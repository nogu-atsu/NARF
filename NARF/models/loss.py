import torch
from torch import nn


class SparseLoss:  # TODO think of a better name
    def __init__(self, config):
        super(SparseLoss, self).__init__()
        self.config = config

        # loss
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def __call__(self, grid, sparse_color, sparse_mask, color, mask=None):
        batchsize, _, size, _ = color.shape
        # TODO: maybe sparse and sampled are confusing
        # TODO: remove these reshape. (these are just added for consistency.)
        color = color.reshape(batchsize, 3, size * size)

        # sample points from image
        sampled_color = torch.gather(color, dim=2, index=grid[:, None].repeat(1, 3, 1))

        if mask is not None:
            foreground_mask = mask.reshape(batchsize, size * size)  # [0, 1] value
            sampled_mask = torch.gather(foreground_mask, dim=1, index=grid)
        else:
            sampled_mask = None

        loss_color, loss_mask = self.img_mask_loss(sampled_color, sparse_color, sampled_mask, sparse_mask)
        return loss_color, loss_mask

    def img_mask_loss(self, real_color, nerf_color, real_mask, nerf_mask):
        if self.config.nerf_loss_type == "mse":
            loss_color = self.mse(real_color, nerf_color) * self.config.color_coef
        elif self.config.nerf_loss_type == "mae":
            def trunc_mae(a, b, thres=0.01):  # mean absolute error
                return torch.clamp_min(torch.abs(a - b), thres).mean()

            loss_color = trunc_mae(real_color, nerf_color) * self.config.color_coef
        else:
            raise ValueError()

        if real_mask is not None:
            # need this loss even if self.config.mask_coef==0 for ddp
            loss_mask = self.mse(real_mask, nerf_mask) * self.config.mask_coef

        else:
            loss_mask = 0

        return loss_color, loss_mask
