import torch
from torch import nn
from torch.nn import init
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weights = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        std = 1
        init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b = 3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weights = nn.Parameter(torch.ones((d_model), **factory_kwargs))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * self.weights / rms

        return result.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We use torch.amax instead of torch.max here
        # as torch.max(input, dim, keepdim=False, *, out=None) returns a named tuple, not Tensor
        x_max = torch.amax(x, dim=self.dim, keepdim=True)
        exp = (x - x_max).exp()
        return exp / exp.sum(dim=self.dim, keepdim=True)

