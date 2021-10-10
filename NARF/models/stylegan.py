# modified from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..stylegan_op import fused_leaky_relu


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel // groups, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel // groups * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConv1d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True,
            bias_init=0, c=1, w=1, init="normal", lr_mul=1
    ):
        super().__init__()
        if init == "normal":
            weight = torch.randn(out_channel, in_channel // groups, kernel_size).div_(lr_mul)
        elif init == "uniform":
            weight = torch.FloatTensor(out_channel, in_channel // groups, kernel_size).uniform_(-1, 1).div_(lr_mul)
        else:
            raise ValueError()
        self.weight = nn.Parameter(weight)
        self.scale = w * c ** 0.5 / math.sqrt(in_channel / groups * kernel_size) * lr_mul

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    @property
    def memory_cost(self):
        return self.out_channel

    @property
    def flops(self):
        f = 2 * self.in_channel * self.out_channel // self.groups - self.out_channel
        if self.bias is not None:
            f += self.out_channel
        return f

    def forward(self, input):
        out = F.conv1d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class NormalizedConv1d(nn.Module):
    # stylegan2 like normalization
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True,
            c=1, w=1, init="normal", lr_mul=1
    ):
        super().__init__()

        self.init = init
        self.scale = w * c ** 0.5 * lr_mul
        if init == "normal":
            self.weight = nn.Parameter(
                torch.randn(out_channel, in_channel // groups, kernel_size)
            )
        elif init == "uniform":
            weight = torch.FloatTensor(out_channel, in_channel // groups, kernel_size).uniform_(-1, 1).div_(lr_mul)
            self.weight = nn.Parameter(weight)
        else:
            raise ValueError()

        self.stride = stride
        self.padding = padding
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

        self.in_channel = in_channel
        self.out_channel = out_channel

    @property
    def memory_cost(self):
        return self.out_channel

    @property
    def flops(self):
        f = 2 * self.in_channel * self.out_channel // self.groups - self.out_channel
        if self.bias is not None:
            f += self.out_channel
        return f

    def forward(self, input):
        scale = self.scale * torch.rsqrt(self.weight.pow(2).sum([1, 2], keepdim=True) + 1e-8)
        if self.init == "uniform":
            scale = scale / 3 ** 0.5  # std of uniform = std of normal / 3**0.5
        out = F.conv1d(
            input,
            self.weight * scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, w=1
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (w / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def memory_cost(self):
        return self.out_dim

    @property
    def flops(self):
        f = 2 * self.in_dim * self.out_dim - self.out_dim
        if self.bias is not None:
            f += self.out_dim
        return f

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            assert self.bias is not None
            bias = self.bias * self.lr_mul
            out = fused_leaky_relu(out, bias)

        else:
            bias = None if self.bias is None else self.bias * self.lr_mul
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
