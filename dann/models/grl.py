import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverseLayer(torch.autograd.Function):
    """
    Extension of gradient reversal layer
    """
    @staticmethod
    def forward(ctx, x, hp_lambda=1.0):
        ctx.hp_lambda = hp_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.hp_lambda
        return grad_output, None

    def grad_reverse(x, hp_lambda=1.0):
        return GradReverseLayer.apply(x, hp_lambda)
