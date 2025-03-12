import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from .weight_init import trunc_normal_

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, drop=0.):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear(x)
        return self.drop(x)

class PositiveAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.rand((in_dim, out_dim)))
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = torch.matmul(x, torch.exp(self.weight))
        return self.drop(x)
    
class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, dim=128, patch_size=(5, 1), in_chans=3):
        super().__init__()
        ph, pw = patch_size
        self.to_patch = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        self.proj = nn.Linear(in_chans * ph * pw, dim)

    def forward(self, x: torch.Tensor):
        x = self.to_patch(x)
        x = self.proj(x)
        return x
    
class PositionEmbed(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
    
    def forward(self, x):
        B, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> B 1 d", B=B)
        x = torch.concat([cls_tokens, x], dim = 1)
        x += self.pos_embedding
        return x