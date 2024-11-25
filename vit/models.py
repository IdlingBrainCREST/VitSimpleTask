import torch
from torch import nn
from .vit import VisionTransformer, PositiveAdapter

class SingleDecoder(nn.Module):
    def __init__(
            self, input_dim, length=50, num_classes=3, dim=32, depth=6, num_heads=8, patch_size=1, 
            mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0.
        ):
        super().__init__()
        self.tr = VisionTransformer(
            img_size=(length, input_dim), patch_size=patch_size, in_chans=1, num_classes=num_classes, 
            embed_dim=dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            mlp_head=mlp_head, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = self.tr(x)
        return x
    
    def pred(self, x) -> torch.Tensor:
        x = self.forward(x)
        return self.sigmoid(x)