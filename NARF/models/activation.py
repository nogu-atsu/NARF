import torch.nn.functional as F
from torch.autograd import Function


class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):  # Avoid zero gradient in the negative region.
        slope = 0.1
        input, = ctx.saved_tensors
        grad_input = grad_output * (input >= 0) + grad_output * (input < 0) * (grad_output < 0) * slope
        return grad_input
