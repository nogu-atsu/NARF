from typing import Tuple

import torch
import numpy as np


def sample_points(batchsize, condition=None, size=128, n=128, coordinate_noise=False):
    rand_condition = torch.cuda.FloatTensor(batchsize, size ** 2).uniform_()
    if condition is not None:
        rand_condition += condition.float()

    val, idx = torch.topk(rand_condition, k=n, dim=1)  # B*numbone x n
    if coordinate_noise:
        x = idx % size + torch.zeros_like(idx, dtype=torch.float32).uniform_()  # + 0.5
        y = idx // size + torch.zeros_like(idx, dtype=torch.float32).uniform_()  # + 0.5
    else:
        x = idx % size + 0.5
        y = idx // size + 0.5
    img_coord = torch.stack([x, y, torch.ones_like(x)], dim=1)  # B*num_bone x 3 x n
    return idx, img_coord.float(), rand_condition >= val.min(dim=1, keepdim=True)[0]


def sample_patch(batchsize=8, size=128, patchsize=24, dilation=1, coordinate_noise=False):
    dilated_patchsize = (patchsize - 1) * dilation + 1
    x = torch.randint(0, size - dilated_patchsize, (batchsize,), device="cuda")
    y = torch.randint(0, size - dilated_patchsize, (batchsize,), device="cuda")

    # calculate patch coordinates
    left_top_idx = y * size + x

    relative_idx = (torch.arange(0, patchsize, dilation, device="cuda") +
                    torch.arange(0, patchsize, dilation, device="cuda")[:, None] * size).reshape(-1)

    idx = left_top_idx[:, None] + relative_idx
    if coordinate_noise:
        x = idx % size + torch.zeros_like(idx, dtype=torch.float32).uniform_()  # + 0.5
        y = idx // size + torch.zeros_like(idx, dtype=torch.float32).uniform_()  # + 0.5
    else:
        x = idx % size + 0.5
        y = idx // size + 0.5
    img_coord = torch.stack([x, y, torch.ones_like(x)], dim=1)  # B x 3 x n
    return idx, img_coord.float()


def sample_points_according_to_mask(foreground_mask, size=128, n=128, coordinate_noise=False, large_number=100):
    # from background
    background_idx, background_img_coord, background_bool = sample_points(
        (foreground_mask < 0.05).float(), size=size, n=n * 2, coordinate_noise=coordinate_noise)
    already_sampled = background_bool

    # from foreground
    foreground_idx, foreground_img_coord, foreground_bool = sample_points(
        (foreground_mask > 0.05).float() - already_sampled.float() * large_number, size=size, n=n,
        coordinate_noise=coordinate_noise)

    # merge points
    sampled_idx = torch.cat([background_idx, foreground_idx], dim=1)
    sampled_img_coord = torch.cat([background_img_coord,
                                   foreground_img_coord], dim=2).float()  # B x 3 x (num_sample * 3)

    return sampled_idx, sampled_img_coord


def flex_grid_ray_sampler(render_size: int, patch_size: int, batchsize: int) -> Tuple[torch.tensor, torch.tensor]:
    """sample rays

    Args:
        render_size:
        patch_size:
        batchsize:

    Returns:
        grid: normalized coordinates used for torch.nn.functional.grid_sample, (B, patch_size, patch_size, 2)
        homo_img: homogeneous coordinate, (B, 1, 3, patch_size ** 2)

    """
    # reimplement FlexGridRaySampler in GRAF [NeurIPS 2020]
    y, x = torch.meshgrid([torch.linspace(-1, 1, patch_size, device="cuda"),
                           torch.linspace(-1, 1, patch_size, device="cuda")])  # [-1, 1]
    grid = torch.stack([x, y], dim=2)[None]
    grid = grid.repeat(batchsize, 1, 1, 1)  # B x patch_size x patch_size x 2

    scale = torch.cuda.FloatTensor(batchsize).uniform_((patch_size - 1) / render_size,
                                                       (render_size - 1) / render_size)

    shift_xy = torch.cuda.FloatTensor(batchsize, 2).uniform_(-1, 1) * (1 - 1 / render_size - scale[:, None])
    grid = grid * scale[:, None, None, None] + shift_xy[:, None, None]

    rays = (grid + 1) * render_size / 2
    rays = rays.reshape(batchsize, -1, 2).permute(0, 2, 1)
    homo_img = torch.cat([rays, torch.ones(batchsize, 1, patch_size ** 2, device="cuda")], dim=1)  # B x 3 x n
    homo_img = homo_img.reshape(batchsize, 1, 3, -1)
    return grid, homo_img


def whole_image_grid_ray_sampler(render_size: int, patch_size: int, batchsize: int
                                 ) -> Tuple[torch.tensor, torch.tensor]:
    """sample rays from entire image

    Args:
        render_size:
        patch_size:
        batchsize:

    Returns:
        grid: normalized coordinates used for torch.nn.functional.grid_sample, (B, patch_size, patch_size, 2)
        homo_img: homogeneous image coordinate, (B, 1, 3, patch_size ** 2)

    """
    y, x = torch.meshgrid([torch.arange(patch_size, device="cuda"),
                           torch.arange(patch_size, device="cuda")])
    rays = torch.stack([x, y], dim=2)[None]
    rays = render_size * (rays + 0.5) / patch_size
    rays = rays.repeat(batchsize, 1, 1, 1)  # B x patch_size x patch_size x 2

    grid = rays / (render_size / 2) - 1  # [-1, 1]

    rays = rays.reshape(batchsize, -1, 2).permute(0, 2, 1)
    homo_img = torch.cat([rays, torch.ones(batchsize, 1, patch_size ** 2, device="cuda")], dim=1)  # B x 3 x n
    homo_img = homo_img.reshape(batchsize, 1, 3, -1)
    return grid, homo_img


def random_ray_sampler(render_size, patch_size, batchsize):
    # sample patch_size**2 random rays
    grid, homo_img, _ = sample_points(batchsize=batchsize, size=render_size, n=patch_size ** 2)
    homo_img = homo_img.reshape(batchsize, 1, 3, -1)
    return grid, homo_img


def random_or_patch_sampler(render_size, patch_size, batchsize):
    # 50% random sample, 50% patch sample
    grid, img_coord = random_ray_sampler(render_size, patch_size, batchsize)

    # sample patch
    patch_grid, patch_img_coord = sample_patch(batchsize=batchsize, size=render_size, patchsize=patch_size)
    patch_img_coord = patch_img_coord.reshape(batchsize, 1, 3, -1)

    # choose random sample or patch sample
    rand = (torch.cuda.FloatTensor(patch_grid.shape[0]).uniform_(0, 1) < 0.5).long()

    grid = patch_grid * rand[:, None] + grid * (1 - rand[:, None])
    img_coord = patch_img_coord * rand[:, None, None, None] + img_coord * (1 - rand[:, None, None, None])
    return grid, img_coord


def in_cube(p: torch.Tensor):
    # whether the positions are in the cube [-1, 1]^3
    # :param p: b x groups * 3 x n (n = num_of_ray * points_on_ray)
    if p.shape[1] == 3:
        inside = (p.abs() <= 1).all(dim=1, keepdim=True)  # ? x 1 x ???
        return inside
    b, _, n = p.shape
    inside = (p.reshape(b, -1, 3, n).abs() <= 1).all(dim=2)
    return inside  # b x groups x 1 x n


def all_reduce(scalar):
    scalar = torch.tensor(scalar).cuda()
    torch.distributed.all_reduce(scalar)
    return scalar.item()


def get_module(model, ddp):
    return model.module if ddp else model


def get_final_papernt_id(parents):
    have_child = [i for i in range(24) if i in parents]
    new_id = {have_child[i]: i for i in range(len(have_child))}
    new_id[-1] = -1
    new_parent = np.array([new_id[parents[have_child[i]]] for i in range(len(have_child))])
    return new_parent
