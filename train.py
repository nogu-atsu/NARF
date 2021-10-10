import argparse
import copy
import json
import os
import time

import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from NARF.dataset import THUmanDataset, BlenderDataset
from NARF.models.loss import SparseLoss
from NARF.models.net import NeRFGenerator, Generator, NeRFAutoEncoder
from NARF.models.tiny_utils import random_ray_sampler, all_reduce, get_module
from NARF.utils import record_setting, yaml_config, write
from NARF.visualization_utils import save_img, ssim, psnr


def train(config, validation=False, ae_test=False):
    if validation:
        dataset, data_loader = create_dataloader(config.dataset, novel_pose_novel_view=True, ae_test=ae_test)
        validation_func(config, dataset, data_loader, rank=0, ddp=False)
    else:
        dataset, data_loader = create_dataloader(config.dataset)
        train_func(config, dataset, data_loader, rank=0, ddp=False)


def create_dataloader(config_dataset, novel_pose_novel_view=False, ae_test=False):
    batch_size = config_dataset.bs
    shuffle = True
    drop_last = True
    num_workers = config_dataset.num_workers

    dataset_train, datasets_val = create_dataset(config_dataset, novel_pose_novel_view=novel_pose_novel_view,
                                                 ae_test=ae_test)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                              drop_last=drop_last)
    val_loaders = {key: DataLoader(datasets_val[key], batch_size=1, num_workers=num_workers, shuffle=False,
                                   drop_last=False) for key in datasets_val.keys()}
    return (dataset_train, datasets_val), (train_loader, val_loaders)


def cache_dataset(config_dataset):
    create_dataset(config_dataset, just_cache=True)


def create_dataset(config_dataset, just_cache=False, novel_pose_novel_view=False,
                   ae_test=False):  # novel_pose_novel_view is temprary
    # TODO: config.dataset
    # TODO: move more args to config
    size = config_dataset.image_size
    dataset_name = config_dataset.name
    if ae_test:
        from easydict import EasyDict as edict
        ae_config = yaml_config("configs/supervised_part_wise_NeRF/THUman/THUman_ae_test.yml",
                                "configs/supervised_part_wise_NeRF/default.yml")
        train_dataset_config = ae_config.dataset.train
        val_dataset_config = ae_config.dataset.val


    elif config_dataset.data_root is not None:
        from easydict import EasyDict as edict
        train_dataset_config = edict({"data_root": config_dataset.data_root,
                                      "train": True,
                                      "n_mesh": 50,
                                      "n_rendered_per_mesh": 100,
                                      "n_imgs_per_mesh": 100})
        val_dataset_config = edict({"novel_pose": {"data_root": config_dataset.data_root,
                                                   "train": False,
                                                   "n_mesh": 6,
                                                   "n_rendered_per_mesh": 100,
                                                   "n_imgs_per_mesh": 20}})
    elif novel_pose_novel_view:
        from easydict import EasyDict as edict
        train_dataset_config = config_dataset.train
        val_dataset_config = edict({"novel_pose_novel_view": {
            "data_root": config_dataset.val.novel_view.data_root,
            "train": False,
            "n_mesh": config_dataset.val.novel_pose.n_mesh,
            "n_rendered_per_mesh": 20,
            "n_imgs_per_mesh": 20}})
    else:
        train_dataset_config = config_dataset.train
        val_dataset_config = config_dataset.val

    print("loading datasets")
    if dataset_name == "human":
        dataset_train = THUmanDataset(train_dataset_config, size=size, return_bone_params=True,
                                      return_bone_mask=True, random_background=False, just_cache=just_cache,
                                      load_camera_intrinsics=config_dataset.load_camera_intrinsics)
        datasets_val = dict()
        for key in val_dataset_config.keys():
            if val_dataset_config[key].data_root is not None:
                datasets_val[key] = THUmanDataset(val_dataset_config[key], size=size, return_bone_params=True,
                                                  return_bone_mask=True, random_background=False, num_repeat_in_epoch=1,
                                                  just_cache=just_cache,
                                                  load_camera_intrinsics=config_dataset.load_camera_intrinsics)
    elif dataset_name == "bulldozer":
        dataset_train = BlenderDataset(train_dataset_config, size=size, return_bone_params=True,
                                       random_background=False, just_cache=just_cache)
        datasets_val = dict()
        for key in val_dataset_config.keys():
            if val_dataset_config[key].data_root is not None:
                datasets_val[key] = BlenderDataset(val_dataset_config[key], size=size, return_bone_params=True,
                                                   random_background=False, num_repeat_in_epoch=1,
                                                   just_cache=just_cache)
    else:
        assert False, f"{dataset_name} is not supported"

    return dataset_train, datasets_val


def validate(gen, val_loaders, config, ddp=False, metric=["SSIM", "PSNR"]):
    mse = nn.MSELoss()

    size = config.dataset.image_size
    nerf_only = config.nerf_only
    auto_encoder = config.auto_encoder

    if auto_encoder:  # load view2 dataset for novel view reconstruction
        datasets_val = dict()
        val_dataset_config = config.dataset.val
        for key in val_dataset_config.keys():
            if val_dataset_config[key].data_root is not None:
                if val_dataset_config[key].data_root.split("/")[-1] == "all_person_novel_view_test":
                    dataset = copy.deepcopy(val_dataset_config[key])
                    dataset["data_root"] = "/home/acc12675ut/data/dataset/THUman/all_person_novel_view_test_view2"
                    datasets_val[key] = THUmanDataset(dataset, size=size, return_bone_params=False,
                                                      return_bone_mask=False, random_background=False,
                                                      num_repeat_in_epoch=1, just_cache=False,
                                                      load_camera_intrinsics=dataset.load_camera_intrinsics)

    loss = dict()

    loss_func = {"L2": mse, "SSIM": ssim, "PSNR": psnr}
    for key, val_loader in val_loaders.items():
        num_data = len(val_loader.dataset)

        val_loss_color = 0
        val_loss_mask = 0
        val_loss_color_metric = {met: 0 for met in metric}
        for i, data in tqdm(enumerate(val_loader)):
            gen.eval()
            with torch.no_grad():
                batch = {key: val.cuda(non_blocking=True).float() for key, val in data.items()}

                img = batch["img"]
                mask = batch["mask"]

                if "disparity" in batch:
                    disparity = batch["disparity"]
                    bone_mask = batch["bone_mask"]
                    part_bone_disparity = batch["part_bone_disparity"]
                    keypoint_mask = batch["keypoint_mask"]

                if "inv_intrinsics" in batch:
                    inv_intrinsics = batch["inv_intrinsics"]
                else:
                    inv_intrinsics = None

                if "pose_to_world" in batch:
                    pose_to_camera = batch["pose_to_camera"]
                    pose_to_world = batch["pose_to_world"]
                    if "bone_length" in batch:
                        bone_length = batch["bone_length"]
                    else:
                        bone_length = None

                if auto_encoder:
                    if key in datasets_val:
                        img = datasets_val[key][i]["img"]
                        img = torch.tensor(img).float().cuda()[None]
                    z = get_module(gen, ddp).encoder(img)
                else:
                    z = None

                if nerf_only:
                    # arf
                    if inv_intrinsics is None:
                        inv_intrinsics = torch.tensor(get_module(gen, ddp).inv_intrinsics).float().cuda()
                    nerf = get_module(gen, ddp).nerf
                    gen_color, gen_mask, _ = nerf.render_entire_img(pose_to_camera,
                                                                    inv_intrinsics, z=z,
                                                                    bone_length=bone_length,
                                                                    world_pose=pose_to_world,
                                                                    thres=0.0,
                                                                    render_size=size,
                                                                    batchsize=2000)
                    if torch.isnan(gen_color).any():
                        print("NaN is detected")
                    gen_color = gen_color[None]
                    gen_mask = gen_mask[None, None]
                    gen_color = gen_color - (1 - gen_mask)
                else:
                    # cnn
                    gen_color, gen_mask = get_module(gen, ddp).cnn_forward(disparity, part_bone_disparity,
                                                                           keypoint_mask,
                                                                           z, pose_to_camera, pose_to_world)
                    gen_mask = torch.sigmoid(gen_mask)

                val_loss_mask += loss_func["L2"](mask, gen_mask).item()
                val_loss_color += loss_func["L2"](img, gen_color).item()

                for met in metric:
                    val_loss_color_metric[met] += loss_func[met](img, gen_color).item()
        if ddp:
            val_loss_mask = all_reduce(val_loss_mask)
            val_loss_color = all_reduce(val_loss_color)

            for met in metric:
                val_loss_color_metric[met] = all_reduce(val_loss_color_metric[met])

        loss[key] = {"color": val_loss_color / num_data,
                     "mask": val_loss_mask / num_data}
        for met in metric:
            loss[key][f"color_{met}"] = val_loss_color_metric[met] / num_data

    return loss


def train_func(config, dataset, data_loader, rank, ddp=False, world_size=1):
    torch.backends.cudnn.benchmark = True

    out_dir = config.out_root
    out_name = config.out
    if rank == 0:
        writer = tbx.SummaryWriter(f"{out_dir}/runs/{out_name}")
        os.makedirs(f"{out_dir}/result/{out_name}", exist_ok=True)
        record_setting(f"{out_dir}/result/{out_name}")

    size = config.dataset.image_size
    num_iter = config.num_iter

    dataset = dataset[0]
    num_bone = dataset.num_bone
    intrinsics = dataset.intrinsics

    if config.auto_encoder:
        if config.nerf_only:
            gen = NeRFAutoEncoder(config.generator_params, size, intrinsics, num_bone, ch=32,
                                  parent_id=dataset.output_parents)
        else:
            gen = Generator(config.generator_params,
                            in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                            num_bone=dataset.num_bone, ch=32, size=size,
                            intrinsics=dataset.cp.intrinsics, rgb=True, auto_encoder=True,
                            parent_id=dataset.output_parents)
    elif config.nerf_only:
        gen = NeRFGenerator(config.generator_params, size, intrinsics, num_bone,
                            ray_sampler=random_ray_sampler,
                            parent_id=dataset.output_parents)
    else:
        gen = Generator(config.generator_params,
                        in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                        num_bone=dataset.num_bone, ch=32, size=size,
                        intrinsics=dataset.cp.intrinsics, rgb=True,
                        parent_id=dataset.output_parents)

    loss_func = SparseLoss(config.loss)

    num_gpus = torch.cuda.device_count()
    n_gpu = rank % num_gpus

    torch.cuda.set_device(n_gpu)
    gen = gen.cuda(n_gpu)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[n_gpu])

    gen_optimizer = optim.Adam(gen.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-12)

    if config.scheduler_gamma < 1:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, config.scheduler_gamma)

    start_time = time.time()
    iter = 0

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")
            if ddp:
                gen_module = gen.module
            else:
                gen_module = gen
            gen_module.load_state_dict(snapshot["gen"], strict=True)
            gen_optimizer.load_state_dict(snapshot["gen_opt"])
            iter = snapshot["iteration"]
            start_time = snapshot["start_time"]
            del snapshot

    train_loader, val_loaders = data_loader

    mse = nn.MSELoss()
    train_loss_color = 0
    train_loss_mask = 0

    accumulated_train_time = 0
    log = {}

    train_start = time.time()

    val_interval = config.val_interval
    while iter < num_iter:
        for i, data in enumerate(train_loader):
            if rank == 0:
                print(iter)
            if (iter + 1) % 100 == 0 and rank == 0:
                print(f"{iter + 1} iter, {(time.time() - start_time) / iter} s/iter")
            gen.train()

            batch = {key: val.cuda(non_blocking=True).float() for key, val in data.items()}
            img = batch["img"]
            mask = batch["mask"]

            if "disparity" in batch:
                disparity = batch["disparity"]
                bone_mask = batch["bone_mask"]
                part_bone_disparity = batch["part_bone_disparity"]
                keypoint_mask = batch["keypoint_mask"]

            if "inv_intrinsics" in batch:
                inv_intrinsics = batch["inv_intrinsics"]
            else:
                inv_intrinsics = None

            if "pose_to_world" in batch:
                pose_to_camera = batch["pose_to_camera"]
                pose_to_world = batch["pose_to_world"]
                if "bone_length" in batch:
                    bone_length = batch["bone_length"]
                else:
                    bone_length = None

            gen_optimizer.zero_grad()
            # generate image (sparse sample)
            if config.auto_encoder:
                if config.nerf_only:
                    nerf_color, nerf_mask, grid = gen(pose_to_camera, pose_to_world,
                                                      bone_length, img, inv_intrinsics=inv_intrinsics)
                    loss_color, loss_mask = loss_func(grid, nerf_color, nerf_mask, img, mask)
                    loss = loss_color + loss_mask
                else:
                    cnn_color, cnn_mask = gen.cnn_forward(disparity, part_bone_disparity, keypoint_mask,
                                                          None, pose_to_camera, pose_to_world, img=img)
                    cnn_mask = torch.sigmoid(cnn_mask)
                    loss_color = mse(img, cnn_color) * config.loss.color_coef
                    loss_mask = mse(mask, cnn_mask) * config.loss.mask_coef
                    loss = loss_color + loss_mask

            elif config.nerf_only:
                nerf_color, nerf_mask, grid = gen(pose_to_camera, pose_to_world, bone_length,
                                                  inv_intrinsics=inv_intrinsics)
                loss_color, loss_mask = loss_func(grid, nerf_color, nerf_mask, img, mask)
                loss = loss_color + loss_mask
            else:  # CNN only
                cnn_color, cnn_mask = gen.cnn_forward(disparity, part_bone_disparity, keypoint_mask,
                                                      None, pose_to_camera, pose_to_world)
                cnn_mask = torch.sigmoid(cnn_mask)
                loss_color = mse(img, cnn_color) * config.loss.color_coef
                loss_mask = mse(mask, cnn_mask) * config.loss.mask_coef
                loss = loss_color + loss_mask
            # accumulate train loss
            train_loss_color += loss_color.item() * config.dataset.bs
            train_loss_mask += loss_mask.item() * config.dataset.bs

            if (iter + 1) % 100 == 0 and rank == 0:  # tensorboard
                write(iter, loss, "gen", writer)
            loss.backward()

            gen_optimizer.step()
            # update selector tmp
            if config.generator_params.nerf_params.selector_adaptive_tmp.gamma != 1:
                get_module(gen, ddp).nerf.update_selector_tmp()

            if config.scheduler_gamma < 1:
                scheduler.step()
            torch.cuda.empty_cache()

            if (iter + 1) % 200 == 0 and rank == 0:
                # save image if cnn_only
                if not config.nerf_only:  # cnn only
                    save_img(cnn_color, f"{out_dir}/result/{out_name}/rgb_{iter // 5000 * 5000}.png")
                    save_img(img, f"{out_dir}/result/{out_name}/real_{iter // 5000 * 5000}.png")
                    save_img(bone_mask, f"{out_dir}/result/{out_name}/bone_{iter // 5000 * 5000}.png")

                if ddp:
                    gen_module = gen.module
                else:
                    gen_module = gen
                save_params = {"iteration": iter,
                               "start_time": start_time,
                               "gen": gen_module.state_dict(),
                               "gen_opt": gen_optimizer.state_dict(),
                               }
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_latest.pth")
                torch.save(save_params, f"{out_dir}/result/{out_name}/snapshot_{(iter // 50000 + 1) * 50000}.pth")
            if (iter + 1) % val_interval == 0:
                # add train time
                accumulated_train_time += time.time() - train_start

                val_loss = validate(gen, val_loaders, config, ddp)
                torch.cuda.empty_cache()

                if ddp:
                    train_loss_color = all_reduce(train_loss_color)
                    train_loss_mask = all_reduce(train_loss_mask)

                train_loss_color = train_loss_color / (val_interval * world_size * config.dataset.bs)
                train_loss_mask = train_loss_mask / (val_interval * world_size * config.dataset.bs)

                # write log
                log_ = {"accumulated_train_time": accumulated_train_time,
                        "train_loss_color": train_loss_color,
                        "train_loss_mask": train_loss_mask}
                for key in val_loss.keys():
                    for metric in val_loss[key].keys():
                        log_[f"val_loss_{key}_{metric}"] = val_loss[key][metric]

                log[iter + 1] = log_

                if rank == 0:
                    with open(f"{out_dir}/result/{out_name}/log.json", "w") as f:
                        json.dump(log, f)

                # initialize train loss
                train_loss_color = 0
                train_loss_mask = 0

                #
                train_start = time.time()

            iter += 1


def validation_func(config, dataset, data_loader, rank, ddp=False):
    out_dir = config.out_root
    out_name = config.out
    size = config.dataset.image_size

    dataset = dataset[0]
    num_bone = dataset.num_bone
    intrinsics = dataset.intrinsics

    if config.auto_encoder:
        if config.nerf_only:
            gen = NeRFAutoEncoder(config.generator_params, size, intrinsics, num_bone, ch=32,
                                  parent_id=dataset.output_parents)
        else:
            gen = Generator(config.generator_params,
                            in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                            num_bone=dataset.num_bone, ch=32, size=size,
                            intrinsics=dataset.cp.intrinsics, rgb=True, auto_encoder=True,
                            parent_id=dataset.output_parents)
    elif config.nerf_only:
        gen = NeRFGenerator(config.generator_params, size, intrinsics, num_bone,
                            ray_sampler=random_ray_sampler,
                            parent_id=dataset.output_parents)
    else:
        gen = Generator(config.generator_params,
                        in_dim=dataset.num_bone + dataset.num_valid_keypoints + 3,
                        num_bone=dataset.num_bone, ch=32, size=size,
                        intrinsics=dataset.cp.intrinsics, rgb=True,
                        parent_id=dataset.output_parents)

    torch.cuda.set_device(rank)
    gen.cuda(rank)

    if ddp:
        gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen)
        gen = nn.parallel.DistributedDataParallel(gen, device_ids=[rank])

    if config.resume or config.resume_latest:
        path = f"{out_dir}/result/{out_name}/snapshot_latest.pth" if config.resume_latest else config.resume
        if os.path.exists(path):
            snapshot = torch.load(path, map_location="cuda")
            if ddp:
                gen_module = gen.module
            else:
                gen_module = gen
            gen_module.load_state_dict(snapshot["gen"], strict=False)
            del snapshot

    train_loader, val_loaders = data_loader

    val_loss = validate(gen, val_loaders, config, ddp, metric=["PSNR", "SSIM"])
    torch.cuda.empty_cache()
    # write log
    if rank == 0:
        with open(f"{out_dir}/result/{out_name}/val_metrics.json", "w") as f:
            json.dump(val_loss, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="NARF/configs/release/default.yml")
    parser.add_argument('--default_config', type=str, default="NARF/configs/release/default.yml")
    parser.add_argument('--resume_latest', action="store_true")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--validation', action="store_true")
    parser.add_argument('--ae_test', action="store_true")
    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, args.resume_latest, args.num_workers)

    train(config, args.validation, ae_test=args.ae_test)
