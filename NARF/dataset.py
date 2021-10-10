import glob
import os
import random

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.utils.data import Dataset
from tqdm import tqdm

from .models.tiny_utils import get_final_papernt_id
from .models.utils_3d import THUmanPrior, CameraProjection, create_mask


class THUmanDataset(Dataset):
    """THUman dataset"""

    def __init__(self, config, size=128, random_background=False, return_bone_params=False,
                 return_bone_mask=False, num_repeat_in_epoch=100, just_cache=False, load_camera_intrinsics=False):
        random.seed()
        self.size = size
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.return_bone_params = return_bone_params
        self.return_bone_mask = return_bone_mask

        # read params from config
        self.data_root = config.data_root
        self.config = config
        self.random_background = config.random_background if hasattr(config, "random_background") else random_background
        self.n_mesh = config.n_mesh
        self.n_rendered_per_mesh = config.n_rendered_per_mesh
        self.n_imgs_per_mesh = config.n_imgs_per_mesh
        self.train = config.train
        self.load_camera_intrinsics = load_camera_intrinsics

        self.just_cache = just_cache

        self.imgs = self.cache_image()
        if self.return_bone_params:
            self.pose_to_world_, self.pose_to_camera_, self.inv_intrinsics_ = self.cache_bone_params()
        if just_cache:
            return

        # [0, 1, ..., n_imgs_per_mesh-1, n_imgs_per_mesh, n_imgs_per_mesh+1, ...]
        data_idx = np.arange(self.n_mesh * self.n_imgs_per_mesh) % self.n_imgs_per_mesh + \
                   np.arange(self.n_mesh * self.n_imgs_per_mesh) // self.n_imgs_per_mesh * self.n_rendered_per_mesh

        if not self.train:
            data_idx = -1 - data_idx  # reverse order
        self.data_idx = data_idx

        self.imgs = self.imgs[data_idx]
        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])
        self.output_parents = get_final_papernt_id(self.parents)

        if self.return_bone_params:
            self.pose_to_world = self.pose_to_world_[data_idx]
            self.pose_to_camera = self.pose_to_camera_[data_idx]
            if self.inv_intrinsics_ is not None:
                self.inv_intrinsics = self.inv_intrinsics_[data_idx]

            self.cp = CameraProjection(size=size)
            self.hpp = THUmanPrior()
            self.num_bone = self.hpp.num_bone
            self.num_bone_param = self.num_bone
            self.num_valid_keypoints = self.hpp.num_valid_keypoints
            self.intrinsics = self.cp.intrinsics

    def cache_image(self):
        if os.path.exists(f"{self.data_root}/render_{self.size}.npy"):
            if self.just_cache:
                return None

            imgs = np.load(f"{self.data_root}/render_{self.size}.npy")

        else:
            imgs = []
            img_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/*.png")
            for path in tqdm(sorted(img_paths)):
                img = cv2.imread(path, -1)
                # img = cv2.resize(img, (self.size, self.size))
                imgs.append(img.transpose(2, 0, 1))

            imgs = np.array(imgs)

            np.save(f"{self.data_root}/render_{self.size}.npy", imgs)

        # load background images
        if "background_path" in self.config and self.config.background_path is not None:
            assert os.path.exists(self.config.background_path)
            self.background_imgs = np.load(self.config.background_path)

        return imgs

    def cache_bone_params(self):
        if os.path.exists(f"{self.data_root}/render_{self.size}_pose_to_world.npy"):
            if self.just_cache:
                return None, None, None

            pose_to_world = np.load(f"{self.data_root}/render_{self.size}_pose_to_world.npy")
            pose_to_camera = np.load(f"{self.data_root}/render_{self.size}_pose_to_camera.npy")

            if self.load_camera_intrinsics:
                inv_intrinsics = np.load(f"{self.data_root}/render_{self.size}_inv_intrinsics.npy")
            else:
                inv_intrinsics = None

        else:
            def save_pose(mode):
                pose = []
                pose_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/pose_to_{mode}_*.npy")
                for path in tqdm(sorted(pose_paths)):
                    pose_ = np.load(path)
                    pose.append(pose_)

                pose = np.array(pose)

                np.save(f"{self.data_root}/render_{self.size}_pose_to_{mode}.npy", pose)
                return pose

            pose_to_world = save_pose("world")
            pose_to_camera = save_pose("camera")

            # camera intrinsics
            if self.load_camera_intrinsics:
                inv_intrinsics = []
                K_paths = glob.glob(f"{self.data_root}/render_{self.size}/color/camera_intrinsics_*.npy")
                for path in tqdm(sorted(K_paths)):
                    K_i = np.load(path)
                    inv_intrinsics.append(np.linalg.inv(K_i))
                inv_intrinsics = np.array(inv_intrinsics)

                np.save(f"{self.data_root}/render_{self.size}_inv_intrinsics.npy", inv_intrinsics)
            else:
                inv_intrinsics = None
        return pose_to_world, pose_to_camera, inv_intrinsics

    def __len__(self):
        return len(self.imgs) * self.num_repeat_in_epoch

    def get_bone_length(self, pose):
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        length = length[[i for i in range(23) if i + 1 not in [2, 3, 13, 14]]]
        return length[:, None]

    @staticmethod
    def remove_blank_part(joint_mat_world, joint_mat_camera):
        idx = [i not in [10, 11, 15, 22, 23] for i in range(24)]
        return joint_mat_world[idx], joint_mat_camera[idx]

    def add_blank_part(self, joint_mat_camera, joint_pos_image):
        idx = [0, 0] + list(range(10)) + [9, 9] + list(range(10, 24))
        return joint_mat_camera[:, idx], joint_pos_image[:, :, idx]

    def __getitem__(self, i):
        i = i % len(self.imgs)

        img = self.imgs[i]

        # bgra -> rgb, a
        mask = img[3:] / 255.
        img = img[:3]

        if self.random_background:
            backrgound = np.ones((3, self.size, self.size)) * np.random.randint(0, 256, size=(3, 1, 1))
            img = img * mask + backrgound * (1 - mask)
        else:
            # blacken background
            img = img * mask
            if hasattr(self, "background_imgs"):
                bg_idx = random.randint(0, len(self.background_imgs) - 1)
                bg = self.background_imgs[bg_idx]
                img = img + bg[::-1] * (1 - mask)

        img = (img / 127.5 - 1).astype("float32")  # 3 x 128 x 128
        img = img[::-1].copy()  # BGR2RGB
        mask = mask.astype("float32")  # 1 x 128 x 128

        return_dict = {"img": img, "mask": mask, "idx": self.data_idx[i]}

        if self.return_bone_params:
            pose_to_world = self.pose_to_world[i]
            pose_to_camera = self.pose_to_camera[i]
            bone_length = self.get_bone_length(pose_to_world)

            if self.return_bone_mask:
                # TODO this flag seems not used
                # intrinsics = np.linalg.inv(self.inv_intrinsics[i]) if self.load_camera_intrinsics else None
                # image_coord = self.cp.pose_to_image_coord(pose_to_camera, intrinsics=intrinsics)
                # pose_to_camera_, image_coord = self.add_blank_part(pose_to_camera[None], image_coord)
                # disparity, bone_mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, pose_to_camera_,
                #                                                                        image_coord,
                #                                                                        self.size)
                disparity, bone_mask, part_bone_disparity, keypoint_mask = [np.array([0], dtype="float32") for _ in
                                                                            range(4)]
                return_dict["disparity"] = disparity
                return_dict["bone_mask"] = bone_mask
                return_dict["part_bone_disparity"] = part_bone_disparity
                return_dict["keypoint_mask"] = keypoint_mask

                if self.load_camera_intrinsics:
                    inv_intrinsics = self.inv_intrinsics[i]  # 3 x 3
                    return_dict["inv_intrinsics"] = inv_intrinsics

            pose_to_world, pose_to_camera = self.remove_blank_part(pose_to_world, pose_to_camera)

            return_dict["pose_to_world"] = pose_to_world.astype("float32")
            return_dict["pose_to_camera"] = pose_to_camera.astype("float32")
            return_dict["bone_length"] = bone_length.astype("float32")
        return return_dict

    def random_sample(self):
        i = random.randint(0, len(self.imgs) - 1)
        return self.__getitem__(i)


class BlenderDataset(THUmanDataset):
    """Blender dataset"""  # TODO THUmanと共通化

    def __init__(self, config, size=200, random_background=False, return_bone_params=False,
                 num_repeat_in_epoch=100, just_cache=False):
        random.seed()
        self.size = size
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.random_background = random_background
        self.return_bone_params = return_bone_params

        # read params from config
        self.data_root = config.data_root
        self.n_mesh = config.n_mesh
        self.n_rendered_per_mesh = config.n_rendered_per_mesh
        self.n_imgs_per_mesh = config.n_imgs_per_mesh
        self.train = config.train
        self.load_camera_intrinsics = False

        self.just_cache = just_cache

        self.imgs = self.cache_image()
        if self.return_bone_params:
            self.pose_to_world, self.pose_to_camera, _ = self.cache_bone_params()
        if just_cache:
            return

        # [0, 1, ..., n_imgs_per_mesh-1, n_imgs_per_mesh, n_imgs_per_mesh+1, ...]
        data_idx = np.arange(self.n_mesh * self.n_imgs_per_mesh) % self.n_imgs_per_mesh + \
                   np.arange(self.n_mesh * self.n_imgs_per_mesh) // self.n_imgs_per_mesh * self.n_rendered_per_mesh

        if not self.train:
            data_idx = -1 - data_idx  # reverse order

        self.imgs = self.imgs[data_idx]

        if self.return_bone_params:
            self.pose_to_world = self.pose_to_world[data_idx]
            self.pose_to_camera = self.pose_to_camera[data_idx]

            self.cp = CameraProjection(size=size)
            # self.hpp = THUmanPrior()
            self.num_bone = 1
            # self.num_valid_keypoints = self.hpp.num_valid_keypoints
            self.intrinsics = self.cp.intrinsics
            self.parents = np.array([-1])
        self.output_parents = np.array([-1])

    def __getitem__(self, i):
        i = i % len(self.imgs)

        img = self.imgs[i]

        # bgra -> rgb, a
        mask = img[3:] / 255.
        img = img[:3]

        if self.random_background:
            backrgound = np.ones((3, self.size, self.size)) * np.random.randint(0, 256, size=(3, 1, 1))
            img = img * mask + backrgound * (1 - mask)
        else:
            # blacken background
            img = img * mask

        img = (img / 127.5 - 1).astype("float32")  # 3 x 128 x 128
        img = img[::-1].copy()  # BGR2RGB
        mask = mask.astype("float32")  # 1 x 128 x 128

        if self.return_bone_params:
            pose_to_world = self.pose_to_world[i][None]
            pose_to_camera = self.pose_to_camera[i][None]

            return img, mask, pose_to_camera, pose_to_world
        else:
            return img, mask

    def random_sample(self):
        i = random.randint(0, len(self.imgs) - 1)
        return self.__getitem__(i)


class THUmanPoseDataset(Dataset):
    def __init__(self, size=128, data_root="", just_cache=False, num_repeat_in_epoch=100):
        self.size = size
        self.data_root = data_root
        self.just_cache = just_cache
        self.num_repeat_in_epoch = num_repeat_in_epoch

        self.poses = self.create_cache()

        self.cp = CameraProjection(size=size)  # just holds camera intrinsics
        self.hpp = THUmanPrior()

        self.num_bone = self.hpp.num_bone
        self.num_bone_param = self.num_bone
        self.num_valid_keypoints = self.hpp.num_valid_keypoints

        self.parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                                 12, 13, 14, 16, 17, 18, 19, 20, 21])
        self.output_parents = get_final_papernt_id(self.parents)

        self.deterministic = False

    def __len__(self):
        return len(self.poses) * self.num_repeat_in_epoch

    def create_cache(self):
        if os.path.exists(f"{self.data_root}/bone_params_128.npy"):
            if self.just_cache:
                return None

            poses = np.load(f"{self.data_root}/bone_params_128.npy")

        else:
            poses = []
            pose_paths = glob.glob(f"{self.data_root}/render_128/bone_params/*.npy")
            for path in pose_paths:
                poses.append(np.load(path))

            poses = np.array(poses)

            np.save(f"{self.data_root}/bone_params_128.npy", poses)

        return poses

    def sample_camera_mat(self, cam_t=None, theta=None, phi=None, angle=None):
        if self.deterministic:
            if cam_t is None:
                cam_t = np.array((0, 0, 2.0))

                theta = 0
                phi = 0
                angle = 0
            cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
            cam_r = cam_r * angle
        else:
            if cam_t is None:
                cam_t = np.array((0, 0, 2.0))

                theta = np.random.uniform(0, 0.3)
                phi = np.random.uniform(0, 2 * np.pi)
                angle = np.random.uniform(0, 2 * np.pi)
            cam_r = np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])
            cam_r = cam_r * angle

        R = cv2.Rodrigues(cam_r)[0]
        T = np.vstack((np.hstack((R, cam_t.reshape(3, 1))), np.zeros((1, 4))))  # 4 x 4

        return T

    def add_blank_part(self, joint_mat_camera, joint_pos_image):
        idx = [0, 0] + list(range(10)) + [9, 9] + list(range(10, 24))
        return joint_mat_camera[:, idx], joint_pos_image[:, :, idx]

    def get_bone_length(self, pose):
        coordinate = pose[:, :3, 3]
        length = np.linalg.norm(coordinate[1:] - coordinate[self.parents[1:]], axis=1)
        length = length[[i for i in range(23) if i + 1 not in [2, 3, 13, 14]]]
        return length[:, None]

    def preprocess(self, pose, scale=0.5):
        left_hip = 1
        right_hip = 2
        trans = -pose[[left_hip, right_hip], :3, 3].mean(axis=0)
        pose_copy = pose.copy()
        pose_copy[:, :3, 3] += trans[None,]
        pose_copy[:, :3, 3] *= scale
        return pose_copy

    def rotate_pose_in_place(self, pose, x_r, y_r, z_r):
        """rotates model (x-axis first, then y-axis, and then z-axis)"""
        mat_x, _ = cv2.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
        mat_y, _ = cv2.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
        mat_z, _ = cv2.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
        mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)
        T = np.eye(4)
        T[:3, :3] = mat

        pose = np.matmul(T, pose)

        return pose

    def transform_randomly(self, pose, scale=True):
        if self.deterministic:
            pose[:, :3, 3] *= 1.3
        else:
            # random rotation
            y_rot = np.random.uniform(-np.pi, np.pi)
            x_rot = np.random.uniform(-0.3, 0.3)
            z_rot = np.random.uniform(-0.3, 0.3)
            # # mul(mat, mesh.v)
            pose = self.rotate_pose_in_place(pose, x_rot, y_rot, z_rot)

            # random scale
            if scale:
                scale = np.random.uniform(1.0, 1.5)
            else:
                scale = 1.3
            pose[:, :3, 3] *= scale
        return pose

    def scale_pose(self, pose, scale):
        pose[:, :3, 3] *= scale
        return pose

    @staticmethod
    def remove_blank_part(joint_mat_world, joint_mat_camera):
        idx = [i not in [10, 11, 15, 22, 23] for i in range(24)]
        return joint_mat_world[idx], joint_mat_camera[idx]

    def __getitem__(self, i):
        i = i % len(self.poses)
        joint_mat_world = self.poses[i]  # 24 x 4 x 4
        joint_mat_world = self.preprocess(joint_mat_world)

        joint_mat_world = self.transform_randomly(joint_mat_world)

        camera_mat = self.sample_camera_mat()

        bone_length = self.get_bone_length(joint_mat_world)

        joint_mat_camera, joint_pos_image = self.cp.process_mat(joint_mat_world, camera_mat)
        joint_mat_camera_, joint_pos_image = self.add_blank_part(joint_mat_camera, joint_pos_image)

        disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_, joint_pos_image,
                                                                          self.size)
        joint_mat_world, joint_mat_camera = self.remove_blank_part(joint_mat_world, joint_mat_camera[0])

        return (disparity,  # size x size
                mask,  # size x size
                part_bone_disparity,  # num_joint x size x size
                joint_mat_camera.astype("float32"),  # num_joint x 4 x 4
                keypoint_mask,  # num_joint x size x size
                bone_length.astype("float32"),  # num_bone x 1
                joint_mat_world.astype("float32"),  # num_joint x 4 x 4
                )

    def batch_same(self, num=100):
        idx = random.randint(0, len(self) - 1)
        d, m, p, j, k, l, w = self[idx]
        data = [d, m, p, j, w, k, l]
        batch = [data for i in range(num)]
        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch

    def batch_sequential_camera_pose(self, num=100, ood=False):  # ood = out of distribution
        joint_mat_world = self.poses[np.random.randint(0, len(self.poses))]  # 24 x 4 x 4
        joint_mat_world = self.preprocess(joint_mat_world)

        joint_mat_world_orig = self.transform_randomly(joint_mat_world)

        batch = []
        for i in range(num):
            if ood:
                camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 1.5 + i / num]),  # 1.5 ~ 2.5
                                                    theta=np.pi * 2 * i / num,
                                                    phi=0,
                                                    angle=np.pi * 2 * i / num)
            else:
                camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 2]), theta=0, phi=0,
                                                    angle=np.pi * 2 * i / num)

            bone_length = self.get_bone_length(joint_mat_world_orig)

            joint_mat_camera, joint_pos_image = self.cp.process_mat(joint_mat_world_orig, camera_mat)
            joint_mat_camera_, joint_pos_image = self.add_blank_part(joint_mat_camera, joint_pos_image)

            disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_,
                                                                              joint_pos_image,
                                                                              self.size)
            joint_mat_world, joint_mat_camera = self.remove_blank_part(joint_mat_world_orig, joint_mat_camera[0])

            batch.append((disparity,  # size x size
                          mask,  # size x size
                          part_bone_disparity,  # num_joint x size x size
                          joint_mat_camera.astype("float32"),  # num_joint x 4 x 4
                          joint_mat_world.astype("float32"),  # num_joint x 4 x 4
                          keypoint_mask,  # num_joint x size x size
                          bone_length.astype("float32"),  # num_bone x 1
                          ))

        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch

    def batch_sequential_bone_param(self, num=100, num_pose=5, num_interpolate_param=None, loop=True,
                                    fix_pose=False, fix_bone_param=False):
        num_parts = 24
        if fix_pose:
            pose_idx = [np.random.randint(0, len(self.poses))] * num_pose
        else:
            pose_idx = np.random.permutation(len(self.poses))[:num_pose]

        joint_mat_world = self.poses[pose_idx]  # num_pose x 24 x 4 x 4
        joint_mat_world = np.stack([self.preprocess(mat) for mat in joint_mat_world])  # num_pose x 24 x 4 x 4

        if fix_bone_param:
            if not fix_pose:
                # joint_mat_world = np.stack([self.transform_randomly(mat, scale=False) for mat
                #                             in joint_mat_world])  # num_pose x 24 x 4 x 4
                joint_mat_world = np.stack([self.scale_pose(mat, 1.3) for i, mat
                                            in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4
        else:
            if fix_pose:
                # # scale all
                # joint_mat_world = np.stack(
                #     [self.scale_pose(mat, 1.25 + np.sin(2 * np.pi * i / num_pose) * 0.25) for i, mat
                #      in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4

                # paper visualization
                def scale_some_parts(pose, idx=12, factor=1.5):
                    t_pa = pose[self.parents[1:], :3, 3]
                    t_ch = pose[1:, :3, 3]
                    t_diff = t_ch - t_pa
                    if isinstance(factor, float):
                        t_diff[idx - 1] *= factor
                    else:
                        t_diff *= factor[:, None]

                    scaled_t = [pose[0, :3, 3]]
                    for i in range(1, pose.shape[0]):
                        scaled_t.append(scaled_t[self.parents[i]] + t_diff[i - 1])
                    scaled_t = np.stack(scaled_t, axis=0)  # num_bone x 3
                    pose[:, :3, 3] = scaled_t
                    return pose  # B x num_bone*3 x 1

                # # paper visualization
                # joint_mat_world = np.stack([self.scale_pose(mat, 1.2) for i, mat
                #                             in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4
                # joint_mat_world = np.stack(
                #     [scale_some_parts(mat, factor=0.9 / 1.2 * i / (num_pose - 1) + 1.8 / 1.2 * (1 - i / (num_pose - 1)))
                #      for i, mat in enumerate(joint_mat_world)])  # num_pose x 24 x 4 x 4

                # git repo
                start_factor = np.random.uniform(1, 1.5, num_parts - 1)
                end_factor = np.random.uniform(1, 1.5, num_parts - 1)
                factor = np.linspace(start_factor, end_factor, num_pose)
                joint_mat_world = np.stack(
                    [scale_some_parts(mat.copy(), factor=factor_i)
                     for factor_i, mat in zip(factor, joint_mat_world)])  # num_pose x 24 x 4 x 4

            else:
                joint_mat_world = np.stack([self.transform_randomly(mat) for mat
                                            in joint_mat_world])  # num_pose x 24 x 4 x 4

        parent_mat = joint_mat_world[:, self.parents[1:]]  # num_pose x 23 x 4 x 4
        parent_mat = np.concatenate([np.tile(np.eye(4)[None, None], (num_pose, 1, 1, 1)), parent_mat], axis=1)

        child_translation = []
        for i in range(num_pose):
            trans_i = []
            for j in range(num_parts):
                trans_i.append(np.linalg.inv(parent_mat[i, j]).dot(joint_mat_world[i, j]))
            child_translation.append(np.array(trans_i))
        child_translation = np.array(child_translation)  # num_pose x 24 x 4 x 4

        # interpolation (slerp)
        interp_pose_to_world = []
        for i in range(num_parts):
            if loop:
                key_rots = np.concatenate([child_translation[:, i, :3, :3],
                                           child_translation[:1, i, :3, :3]], axis=0)  # repeat first
                key_times = np.arange(num_pose + 1)
                times = np.arange(num) * num_pose / num
                interp_trans = np.concatenate([
                    np.linspace(child_translation[j, i, :3, 3],
                                child_translation[(j + 1) % num_pose, i, :3, 3],
                                num // num_pose, endpoint=False) for j in range(num_pose)], axis=0)  # num x 3
            else:
                key_rots = child_translation[:, i, :3, :3]
                key_times = np.arange(num_pose)
                times = np.arange(num) * (num_pose - 1) / (num - 1)
                interp_trans = np.concatenate([
                    np.linspace(child_translation[j, i, :3, 3],
                                child_translation[(j + 1), i, :3, 3],
                                num // (num_pose - 1), endpoint=True) for j in range(num_pose - 1)], axis=0)  # num x 3
            slerp = Slerp(key_times, R.from_matrix(key_rots))
            interp_rots = slerp(times).as_matrix()  # num x 3 x 3

            interp_mat = np.concatenate([interp_rots, interp_trans[:, :, None]], axis=2)
            interp_mat = np.concatenate([interp_mat, np.tile(np.array([[[0, 0, 0, 1]]]), (num, 1, 1))],
                                        axis=1)  # num x 4 x 4
            interp_pose_to_world.append(interp_mat)
        interp_pose_to_world = np.array(interp_pose_to_world)  # num_parts x num x 4 x 4

        # fixed camera
        camera_mat = self.sample_camera_mat(cam_t=np.array([0, 0, 2]), theta=0, phi=0, angle=0)

        batch = []
        for i in range(num):
            mixed_pose_to_world = []
            for part_idx in range(num_parts):
                if self.parents[part_idx] == -1:
                    mat = np.eye(4)
                else:
                    mat = mixed_pose_to_world[self.parents[part_idx]]
                mat = mat.dot(interp_pose_to_world[part_idx, i])

                mixed_pose_to_world.append(mat)

            mixed_pose_to_world_orig = np.stack(mixed_pose_to_world)

            bone_length = self.get_bone_length(mixed_pose_to_world_orig)

            joint_mat_camera, joint_pos_image = self.cp.process_mat(mixed_pose_to_world_orig, camera_mat)
            joint_mat_camera_, joint_pos_image = self.add_blank_part(joint_mat_camera, joint_pos_image)

            disparity, mask, part_bone_disparity, keypoint_mask = create_mask(self.hpp, joint_mat_camera_,
                                                                              joint_pos_image,
                                                                              self.size)
            mixed_pose_to_world, joint_mat_camera = self.remove_blank_part(mixed_pose_to_world_orig,
                                                                           joint_mat_camera[0])

            batch.append((disparity,  # size x size
                          mask,  # size x size
                          part_bone_disparity,  # num_joint x size x size
                          joint_mat_camera.astype("float32"),  # num_joint x 4 x 4
                          mixed_pose_to_world.astype("float32"),  # num_joint x 4 x 4
                          keypoint_mask,  # num_joint x size x size
                          bone_length.astype("float32"),  # num_bone x 1
                          ))

        batch = zip(*batch)
        batch = (torch.tensor(np.stack(b)) for b in batch)
        return batch
