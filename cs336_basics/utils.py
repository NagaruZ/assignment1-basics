import torch
from torch import nn, Tensor
from torch.nn import init
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int

def get_compatible_dff(d_model: int) -> int:
    raw = (8 * d_model) / 3
    return int((d_model + 32) // 64) * 64
