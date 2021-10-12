# Refer to https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .nerf_model import NeRF, PoseConditionalNeRF
from .stylegan import EqualConv2d, EqualLinear
from .model_utils import flex_grid_ray_sampler, random_or_patch_sampler


class ConditionalBatchNorm(nn.Module):
    def __init__(self, ch, z_dim=256, num_bone=2, input_world_pose=False):
        super(ConditionalBatchNorm, self).__init__()
        self.ch = ch
        self.num_bone = num_bone
        self.input_world_pose = input_world_pose

        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.attention = EqualConv2d(ch, num_bone + 1, 3, 1, 1)
        if z_dim == 0:
            self.gamma_background = nn.Parameter(torch.ones(1, 1, self.ch))
            self.beta_background = nn.Parameter(torch.zeros(1, 1, self.ch))
        else:
            self.gamma_z = EqualLinear(z_dim, ch * (num_bone + 1))
            self.beta_z = EqualLinear(z_dim, ch * (num_bone + 1))
        self.gamma_j = EqualLinear(12, ch)
        self.beta_j = EqualLinear(12, ch)
        if input_world_pose:
            self.gamma_w = EqualLinear(12, ch)
            self.beta_w = EqualLinear(12, ch)

    def forward(self, x, z, j, w=None):
        assert w is not None or not self.input_world_pose
        attention = torch.softmax(self.attention(x), dim=1)  # B x (num_bone + 1) x h x w

        num_condition = 1  # how many conditions there are
        j = j[:, :, :3].reshape(-1, 12)
        gamma_foreground = self.gamma_j(j).reshape(-1, self.num_bone, self.ch)
        beta_foreground = self.beta_j(j).reshape(-1, self.num_bone, self.ch)

        if z is not None:
            num_condition += 1
            gamma_z = self.gamma_z(z).reshape(-1, self.num_bone + 1, self.ch)
            beta_z = self.beta_z(z).reshape(-1, self.num_bone + 1, self.ch)

            gamma_foreground = gamma_foreground + gamma_z[:, 1:]
            beta_foreground = beta_foreground + beta_z[:, 1:]

            gamma_background = gamma_z[:, :1]
            beta_background = beta_z[:, :1]
        else:
            batchsize = x.shape[0]
            gamma_background = self.gamma_background.repeat(batchsize, 1, 1)
            beta_background = self.beta_background.repeat(batchsize, 1, 1)

        if self.input_world_pose:
            num_condition += 1
            w = w[:, :, :3].reshape(-1, 12)
            gamma_w = self.gamma_w(w).reshape(-1, self.num_bone, self.ch)
            beta_w = self.beta_w(w).reshape(-1, self.num_bone, self.ch)

            gamma_foreground = gamma_foreground + gamma_w
            beta_foreground = beta_foreground + beta_w

        gamma_foreground = gamma_foreground / num_condition
        beta_foreground = beta_foreground / num_condition

        gamma = torch.cat([gamma_background, gamma_foreground], dim=1)
        beta = torch.cat([beta_background, beta_foreground], dim=1)

        gamma = (gamma[:, :, :, None, None] * attention[:, :, None]).sum(dim=1)  # B x ch x h x w
        beta = (beta[:, :, :, None, None] * attention[:, :, None]).sum(dim=1)  # B x ch x h x w
        return self.bn(x) * (1 + gamma) + beta


class CBNResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1, num_bone=2, z_dim=256, down=False,
                 up=False,
                 input_world_pose=False):
        super(CBNResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down = down
        self.up = up
        if in_channel != out_channel:
            # left
            self.conv_l1 = EqualConv2d(in_channel, out_channel, 1, )

        # right
        self.conv_r1 = EqualConv2d(in_channel, out_channel, conv_size, padding=padding_size, bias=False)
        self.conv_r2 = EqualConv2d(out_channel, out_channel, conv_size, padding=padding_size)

        self.bn1 = ConditionalBatchNorm(out_channel, z_dim=z_dim, num_bone=num_bone, input_world_pose=input_world_pose)
        self.bn2 = ConditionalBatchNorm(out_channel, z_dim=z_dim, num_bone=num_bone, input_world_pose=input_world_pose)

    def forward(self, x, z, j, w=None):
        x = self.relu(x)
        res = x

        # left
        if self.up:
            res = self.upsample(res)
        if self.in_channel != self.out_channel:
            res = self.conv_l1(res)
        if self.down:
            res = self.avg_pool2d(res)

        # right
        if self.up:
            x = self.upsample(x)
        out = self.conv_r1(x)
        out = self.bn1(out, z, j, w)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        if self.down:
            out = self.avg_pool2d(out)

        # merge
        out = res + out
        out = self.bn2(out, z, j, w)

        return out


class UNetRGBGenerator(nn.Module):
    def __init__(self, in_dim=3, num_bone=2, z_dim=256, ch=64, size=128, max_ch=256):
        super(UNetRGBGenerator, self).__init__()
        self.num_down_sample = int(math.log2(size // 4))
        self.num_bone = num_bone
        self.z_dim = z_dim
        self.size = size
        self.conv_mask = EqualConv2d(in_dim, ch, 3, 1, 1)

        self.down_blocks = nn.ModuleList([CBNResBlock(in_channel=min(max_ch, ch * 2 ** i),
                                                      out_channel=min(max_ch, ch * 2 ** (i + 1)), num_bone=num_bone,
                                                      z_dim=z_dim, down=True)
                                          for i in range(self.num_down_sample)])
        self.blocks = nn.ModuleList([CBNResBlock(in_channel=min(max_ch, ch * 2 ** self.num_down_sample),
                                                 out_channel=min(max_ch, ch * 2 ** self.num_down_sample),
                                                 num_bone=num_bone, z_dim=z_dim)
                                     for i in range(1)])
        self.mask_depth_blocks = nn.ModuleList([CBNResBlock(in_channel=min(max_ch, ch * 2 ** (i + 1)) * 2,
                                                            out_channel=min(max_ch, ch * 2 ** i), num_bone=num_bone,
                                                            z_dim=z_dim, up=True)
                                                for i in reversed(range(self.num_down_sample))])
        self.color_blocks = nn.ModuleList([CBNResBlock(in_channel=min(max_ch, ch * 2 ** (i + 1)) * 2,
                                                       out_channel=min(max_ch, ch * 2 ** i), num_bone=num_bone,
                                                       z_dim=z_dim, up=True, input_world_pose=True)
                                           for i in reversed(range(self.num_down_sample))])
        self.mask_conv = EqualConv2d(ch, 1, 3, 1, 1)
        self.color_conv = EqualConv2d(ch, 3, 3, 1, 1)

    def forward(self, bone_disparity, part_disparity, keypoint, z, j, w, background=None, tmp=1.0):
        batchsize = j.shape[0]
        xy = torch.arange(self.size, device="cuda").float() / self.size
        xy = torch.meshgrid(xy, xy)
        xy = torch.stack(xy, dim=0)[None].repeat((batchsize, 1, 1, 1))

        # generate images from bones
        bone_mask = torch.cat([bone_disparity[:, None], (part_disparity > 0.1).float(), keypoint, xy], dim=1)

        if z is None:
            z_mask, z_color = None, None
        else:
            z_mask, z_color = torch.split(z, [self.z_dim, self.z_dim], dim=1)
        h = self.conv_mask(bone_mask)

        down_outs = [h]
        for block in self.down_blocks:
            h = block(h, z_mask, j)
            down_outs.append(h)

        for block in self.blocks:
            h = block(h, z_mask, j)

        h_mask_depth = h
        for i, block in enumerate(self.mask_depth_blocks):
            h_mask_depth = block(torch.cat([h_mask_depth, down_outs[-i - 1]], dim=1), z_mask, j)

        h_color = h
        for i, block in enumerate(self.color_blocks):
            h_color = block(torch.cat([h_color, down_outs[-i - 1]], dim=1), z_color, j, w)

        foreground_mask_logit = self.mask_conv(torch.relu(h_mask_depth)) - 5
        rgb = self.color_conv(torch.relu(h_color))

        foreground_mask = torch.sigmoid(foreground_mask_logit)

        if background is None:
            background = -1
        rgb = rgb * foreground_mask + background * (1 - foreground_mask)  # merge background

        return rgb, foreground_mask_logit


class Generator(nn.Module):
    def __init__(self, config, in_dim=3, num_bone=19, ch=64, size=128, intrinsics=None, rgb=True,
                 ray_sampler=random_or_patch_sampler, auto_encoder=False, parent_id=None):
        """CNN based model

        :param config:
        :param in_dim:
        :param num_bone:
        :param ch:
        :param size:
        :param intrinsics:
        :param rgb:
        :param ray_sampler:
        :param auto_encoder:
        :param parent_id:
        """
        super(Generator, self).__init__()
        assert intrinsics is not None
        self.config = config
        self.size = size
        self.auto_encoder = auto_encoder

        if auto_encoder:
            self.encoder = Encoder(3, config.z_dim, ch=ch)

        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        normalized_intrinsics = np.concatenate([intrinsics[:2] / size, np.array([[0, 0, 1]])], axis=0)
        self.normalized_inv_intrinsics = np.linalg.inv(normalized_intrinsics)
        self.ray_sampler = ray_sampler
        self.cnn_gen = UNetRGBGenerator(in_dim, num_bone, z_dim=config.z_dim, ch=ch, size=size)

    @property
    def memory_cost(self):
        return 0

    @property
    def flops(self):
        return 0

    def forward(self, bone_disparity, part_disparity, keypoint, z, pose_to_camera, pose_to_world, bone_length,
                background, tmp=5.0, img=None, inv_intrinsics=None):
        """
        generate image from 3d bone mask
        :param bone_mask:
        :param z:
        :param j: camera coordinate of joint
        :param w: wold coordinate of joint
        :param background:
        :param tmp:
        :param inv_intrinsics:
        :return:
        """
        if img is not None and self.auto_encoder:
            z = self.encoder(img)

        cnn_color, cnn_mask = self.cnn_gen(bone_disparity, part_disparity, keypoint, z, pose_to_camera,
                                           pose_to_world, background, tmp=tmp)

        return cnn_color, cnn_mask, None, None, None

    def cnn_forward(self, bone_disparity, part_disparity, keypoint, z, pose_to_camera, pose_to_world,
                    background=None, tmp=5.0, img=None):
        """
        generate image from 3d bone mask
        :param bone_mask:
        :param z:
        :param j: camera coordinate of joint
        :param w: wold coordinate of joint
        :param background:
        :param tmp:
        :return:
        """

        if img is not None and self.auto_encoder:
            z = self.encoder(img)
        cnn_color, cnn_mask = self.cnn_gen(bone_disparity, part_disparity, keypoint, z, pose_to_camera,
                                           pose_to_world, background, tmp=tmp)
        return cnn_color, cnn_mask


class NeRFGenerator(nn.Module):
    def __init__(self, config, size, intrinsics=None, num_bone=1, ray_sampler=flex_grid_ray_sampler,
                 parent_id=None):
        super(NeRFGenerator, self).__init__()
        self.config = config
        self.size = size
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        normalized_intrinsics = np.concatenate([intrinsics[:2] / size, np.array([[0, 0, 1]])], axis=0)
        self.normalized_inv_intrinsics = np.linalg.inv(normalized_intrinsics)
        self.num_bone = num_bone
        self.ray_sampler = ray_sampler

        nerf = get_nerf_module(config)
        self.nerf = nerf(config.nerf_params, z_dim=config.z_dim, num_bone=num_bone, bone_length=True,
                         parent=parent_id)

    @property
    def memory_cost(self):
        return self.nerf.memory_cost

    @property
    def flops(self):
        return self.nerf.flops

    def forward(self, pose_to_camera, pose_to_world, bone_length, background=None, z=None, inv_intrinsics=None):
        """
        generate image from 3d bone mask
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length:
        :param background:
        :param z: latent vector
        :param inv_intrinsics:
        :return:
        """
        assert self.num_bone == 1 or (bone_length is not None and pose_to_camera is not None)
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size

        grid, img_coord = self.ray_sampler(self.size, patch_size, batchsize)

        # sparse rendering
        if inv_intrinsics is None:
            inv_intrinsics = self.inv_intrinsics
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(img_coord.device)
        rendered_color, rendered_mask = self.nerf(batchsize, patch_size ** 2, img_coord,
                                                  pose_to_camera, inv_intrinsics, z,
                                                  pose_to_world, bone_length, thres=0.0,
                                                  Nc=self.config.nerf_params.Nc,
                                                  Nf=self.config.nerf_params.Nf)
        if self.ray_sampler in [flex_grid_ray_sampler]:  # TODO unify the way to sample
            rendered_color = rendered_color.reshape(batchsize, 3, patch_size, patch_size)
            rendered_mask = rendered_mask.reshape(batchsize, patch_size, patch_size)

        if background is None:
            rendered_color = rendered_color + (-1) * (1 - rendered_mask[:, None])  # background is black
        else:
            if np.isscalar(background):
                sampled_background = background
            else:
                if self.ray_sampler in [flex_grid_ray_sampler]:  # TODO unify the way to sample
                    sampled_background = torch.nn.functional.grid_sample(background,
                                                                         grid, mode='bilinear')
                else:
                    sampled_background = torch.gather(background, dim=2, index=grid[:, None].repeat(1, 3, 1))

            rendered_color = rendered_color + sampled_background * (1 - rendered_mask[:, None])

        return rendered_color, rendered_mask, grid


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1, down=False,
                 up=False):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down = down
        self.up = up
        if in_channel != out_channel:
            # left
            self.conv_l1 = EqualConv2d(in_channel, out_channel, 1, )

        # right
        self.conv_r1 = EqualConv2d(in_channel, out_channel, conv_size, padding=padding_size, bias=False)
        self.conv_r2 = EqualConv2d(out_channel, out_channel, conv_size, padding=padding_size)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.relu(x)
        res = x

        # left
        if self.up:
            res = self.upsample(res)
        if self.in_channel != self.out_channel:
            res = self.conv_l1(res)
        if self.down:
            res = self.avg_pool2d(res)

        # right
        if self.up:
            x = self.upsample(x)
        out = self.conv_r1(x)
        out = self.bn1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        if self.down:
            out = self.avg_pool2d(out)

        # merge
        out = res + out
        out = self.bn2(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, ch=32):
        """
        Encoder for autoencoder
        :param in_dim: Input channel. Usually 3.
        :param out_dim: Dimension of latent vectors
        :param ch: Hidden size
        """
        super(Encoder, self).__init__()
        self.out_dim = out_dim
        resnet = [EqualConv2d(in_dim, ch, 3, 1, 1),  # out size 128
                  ResBlock(ch, ch * 2, down=True),  # out size 64
                  ResBlock(ch * 2, ch * 4, down=True),  # out size 32
                  ResBlock(ch * 4, ch * 8, down=True),  # out size 16
                  ResBlock(ch * 8, ch * 16, down=True),  # out size 8
                  ResBlock(ch * 16, ch * 16, down=True)]  # out size 4
        self.resnet = nn.Sequential(*resnet)

        self.linear = EqualLinear(4 * 4 * ch * 16, out_dim * 2)

    def __call__(self, x):
        x = self.resnet(x)
        x = F.relu(x, inplace=True)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x  # B x 2*out_dim


def get_nerf_module(config):
    if config.nerf_params.concat_pose:
        return PoseConditionalNeRF
    else:
        return NeRF


class NeRFAutoEncoder(nn.Module):
    def __init__(self, config, size, intrinsics, num_bone=19, ch=32, ray_sampler=random_or_patch_sampler,
                 parent_id=None):
        super(NeRFAutoEncoder, self).__init__()
        assert intrinsics is not None
        self.config = config
        self.size = size

        self.encoder = Encoder(3, config.z_dim, ch=ch)
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        normalized_intrinsics = np.concatenate([intrinsics[:2] / size, np.array([[0, 0, 1]])], axis=0)
        self.normalized_inv_intrinsics = np.linalg.inv(normalized_intrinsics)
        self.ray_sampler = ray_sampler

        nerf = get_nerf_module(config)
        self.nerf = nerf(config.nerf_params, z_dim=config.z_dim, num_bone=num_bone, bone_length=True,
                         parent=parent_id)

    def forward(self, pose_to_camera, pose_to_world, bone_length, img=None, z=None, background=None, tmp=5.0,
                inv_intrinsics=None):
        """
        generate image from 3d bone mask
        :param img:
        :param z:
        :param pose_to_camera: camera coordinate of joint
        :param pose_to_world: wold coordinate of joint
        :param bone_length: bone length
        :param background:
        :param tmp:
        :param inv_intrinsics:
        :return:
        """
        assert img is not None or z is not None
        batchsize = pose_to_camera.shape[0]
        patch_size = self.config.patch_size

        if img is not None:
            z = self.encoder(img)

        grid, img_coord = self.ray_sampler(self.size, patch_size, batchsize)

        # sparse rendering
        if inv_intrinsics is None:
            inv_intrinsics = self.inv_intrinsics
        inv_intrinsics = torch.tensor(inv_intrinsics).float().cuda(img_coord.device)
        nerf_color, nerf_mask = self.nerf(batchsize, patch_size ** 2, img_coord,
                                          pose_to_camera, inv_intrinsics, z,
                                          pose_to_world, bone_length, thres=0.0,
                                          Nc=self.config.nerf_params.Nc,
                                          Nf=self.config.nerf_params.Nf)

        # merge with background
        if background is None:
            nerf_color = nerf_color + (-1) * (1 - nerf_mask[:, None])  # background is black
        else:
            if np.isscalar(background):
                sampled_background = background
            else:
                background = background.reshape(batchsize, 3, self.size ** 2)
                sampled_background = torch.gather(background, dim=2, index=grid[:, None].repeat(1, 3, 1))
            nerf_color = nerf_color + sampled_background * (1 - nerf_mask[:, None])

        return nerf_color, nerf_mask, grid

    def render_entire_img(self, pose_to_camera, bone_length, pose_to_world, z=None, img=None, ):
        assert z is not None or img is not None
        if img is not None:
            z = self.encoder(img)

        inv_intrinsics = torch.tensor(self.inv_intrinsics).float().cuda()
        color, mask, disparity = self.nerf.render_entire_img(pose_to_camera,
                                                             inv_intrinsics, z,
                                                             bone_length=bone_length,
                                                             world_pose=pose_to_world,
                                                             thres=0.0,
                                                             render_size=self.size,
                                                             batchsize=2000, )
        return color, mask, disparity
