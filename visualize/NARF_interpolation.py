# interpolation video
import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
from NARF.dataset import THUmanPoseDataset, THUmanDataset
from NARF.models.net import NeRFGenerator, NeRFAutoEncoder
from NARF.utils import yaml_config


def save_interpolation_video(name="interpolate_camera", segmentation=False, num_frames=30, save_video=True):
    nerf_video = []
    segmentation_video = []
    in_img = []

    is_auto_encoder = hasattr(config, "auto_encoder") and config.auto_encoder
    is_inter_z = (name == "interpolate_z")

    # dataset for autoencoder
    if is_auto_encoder:
        image_dataset = THUmanDataset(config.dataset.val.novel_pose, size=size, return_bone_params=False,
                                      random_background=False)

    for _ in range(num_render ** 2):
        nerf_imgs = []
        segmentation_imgs = []
        if is_auto_encoder:
            def sample_z():
                input_data = image_dataset.random_sample()
                img, mask = input_data["img"], input_data["mask"]
                in_img.append((img + (1 - mask)).transpose(1, 2, 0) * 127.5 + 127.5)
                z = gen.encoder(torch.tensor(img[None]).float().cuda())
                return z

            if is_inter_z:
                zs = [sample_z() for _ in range(num_z)]
            else:
                z = sample_z()
        elif config.generator_params.z_dim == 0:
            z = None
        else:
            z = torch.cuda.FloatTensor(1, config.generator_params.z_dim * 2).normal_()

        if name == "interpolate_camera":
            sequential_batch = dataset.batch_sequential_camera_pose(num=num_frames)
        elif name == "interpolate_z":
            sequential_batch = dataset.batch_same(num=num_frames)
        elif name == "interpolate_camera_ood":
            sequential_batch = dataset.batch_sequential_camera_pose(num=num_frames, ood=True)
        elif name == "interpolate_pose":
            sequential_batch = dataset.batch_sequential_bone_param(num=num_frames, num_pose=3, loop=True,
                                                                   fix_bone_param=True)
        elif name == "interpolate_bone":
            sequential_batch = dataset.batch_sequential_bone_param(num=num_frames, num_pose=3, loop=True,
                                                                   fix_pose=True)
        else:
            raise ValueError()
        sequential_batch = [data.cuda() for data in sequential_batch]

        d, m, p, j, w, k, l = sequential_batch

        for i in tqdm(range(num_frames)):
            with torch.no_grad():
                if is_inter_z:
                    # data idx to interpolate
                    assert num_frames % num_z == 0
                    num_frame_for_each = num_frames // num_z
                    idx1 = i // num_frame_for_each
                    idx2 = (idx1 + 1) % num_z
                    eps = (i % num_frame_for_each) / num_frame_for_each
                    z = zs[idx1] * (1 - eps) + zs[idx2] * eps

                    # # paper z interpolation
                    # eps = i / (num_frames - 1)
                    # z = zs[0] * (1 - eps) + zs[1] * eps

                normalized_inv_intrinsics = torch.tensor(gen.normalized_inv_intrinsics).float().cuda()
                (rendered_color, rendered_mask,
                 rendered_disparity) = nerf.render_entire_img(j[i:i + 1],
                                                              normalized_inv_intrinsics,
                                                              z=z,
                                                              bone_length=l[i:i + 1],
                                                              world_pose=w[i:i + 1],
                                                              thres=0.0,
                                                              render_size=render_size,
                                                              batchsize=1000,
                                                              use_normalized_intrinsics=True)
                if segmentation:
                    segmentation_color, segmentation_mask, _ = nerf.render_entire_img(j[i:i + 1],
                                                                                      normalized_inv_intrinsics, z=z,
                                                                                      bone_length=l[i:i + 1],
                                                                                      world_pose=w[i:i + 1],
                                                                                      thres=0.0,
                                                                                      render_size=render_size,
                                                                                      batchsize=1000,
                                                                                      semantic_map=True,
                                                                                      use_normalized_intrinsics=True)

            nerf_img = rendered_color + (1 - rendered_mask) * 1
            nerf_img = nerf_img.cpu().numpy() * 127.5 + 127.5
            nerf_img = nerf_img.transpose(1, 2, 0).reshape(render_size, render_size, 3)
            nerf_img = np.clip(nerf_img, 0, 255).astype("uint8")
            nerf_imgs.append(nerf_img)
            if segmentation:
                segmentation_img = segmentation_color + (1 - segmentation_mask) * 1
                segmentation_img = segmentation_img.cpu().numpy() * 127.5 + 127.5
                segmentation_img = segmentation_img.transpose(1, 2, 0).reshape(render_size, render_size, 3)
                segmentation_img = np.clip(segmentation_img, 0, 255).astype("uint8")
                segmentation_imgs.append(segmentation_img)

        nerf_video.append(nerf_imgs)
        if segmentation:
            segmentation_video.append(segmentation_imgs)

    def save_video_from_img(video, name="inter_camera"):
        frame_rate = 15.0  # frame rate
        img_size = (render_size * num_render, render_size * num_render)  # size of video frame

        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
        writer = cv2.VideoWriter(f'{config.out_root}/result/{config.out}/{name}.mp4', fmt, frame_rate, img_size)

        frames = []
        for imgs in zip(*video):
            frame = np.stack(imgs).reshape(num_render, num_render, render_size, render_size, 3)
            frame = np.concatenate(frame, axis=1)
            frame = np.concatenate(frame, axis=1)
            frames.append(frame)
        for _ in range(1):
            for i in range(num_frames):
                writer.write(frames[i][:, :, ::-1])

        writer.release()

    def save_img(video, name="inter_camera"):
        frames = []
        for imgs in zip(*video):
            frame = np.stack(imgs).reshape(num_render, num_render, render_size, render_size, 3)
            frame = np.concatenate(frame, axis=1)
            frame = np.concatenate(frame, axis=1)
            frames.append(frame)

        frames = [np.pad(frame, ((size // 8, size // 8), (size // 8, size // 8), (0, 0)), constant_values=255) for frame
                  in frames]
        frames = np.concatenate(frames, axis=1)
        frames = Image.fromarray(frames)
        frames.save(f'{config.out_root}/result/{config.out}/{name}.png')

    if save_video:
        save_video_from_img(nerf_video, name=name)
        if segmentation:
            save_video_from_img(segmentation_video, name=name + "_segmentation")
    else:
        save_img(nerf_video, name=name)
        if segmentation:
            save_img(segmentation_video, name=name + "_segmentation")

    # save img
    if is_auto_encoder:
        in_img = np.stack(in_img).reshape(num_render, num_render, size, size, 3)
        in_img = in_img.transpose(1, 2, 0).astype("uint8")
        cv2.imwrite(f'{config.out_root}/result/{config.out}/in_img.png', in_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="../NARF/configs/default.yml")
    parser.add_argument('--default_config', type=str, default="../NARF/configs/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    batchsize = 1
    num_render = 2  # A grid of num_render x num_render videos is generated.
    num_z = 3
    render_size = 256

    config_path = args.config
    default_config_path = args.default_config

    config = yaml_config(config_path, default_config_path)

    out_dir = config.out_root
    out_name = config.out
    size = config.dataset.image_size
    dataset_name = config.dataset.name
    data_root = config.dataset.train.data_root
    random_background = config.dataset.random_background
    dataset = THUmanPoseDataset(size=size, data_root=data_root)
    loader = DataLoader(dataset, batch_size=batchsize, num_workers=1, shuffle=True, drop_last=True)

    cnn_based = config.generator_params.cnn_based
    if hasattr(config, "auto_encoder") and config.auto_encoder:
        gen = NeRFAutoEncoder(config.generator_params, size, dataset.cp.intrinsics,
                              dataset.num_bone, ch=32).to("cuda")

    elif cnn_based:
        assert False, "Please run CNN_interpolation.py"
    else:
        gen = NeRFGenerator(config.generator_params, size, dataset.cp.intrinsics,
                            num_bone=dataset.num_bone).to("cuda")
    nerf = gen.nerf
    gen.load_state_dict(torch.load(f"{config.out_root}/result/{config.out}/snapshot_latest.pth")["gen"], strict=False)
    gen.eval()
    num_bone = dataset.num_bone

    save_video = True
    save_interpolation_video(name="interpolate_camera", segmentation=True, num_frames=30, save_video=save_video)
    save_interpolation_video(name="interpolate_camera_ood", num_frames=90, save_video=save_video)
    save_interpolation_video(name="interpolate_pose", num_frames=30, save_video=save_video)
    save_interpolation_video(name="interpolate_bone", num_frames=30, save_video=save_video)
    if hasattr(config, "auto_encoder") and config.auto_encoder:
        save_interpolation_video(name="interpolate_z", num_frames=30, save_video=save_video)
