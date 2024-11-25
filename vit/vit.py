import torch
from torch import nn
from einops import rearrange
from .weight_init import trunc_normal_

class Add(nn.Module):
    def forward(self, inputs):
        return torch.add(*inputs)
    
class Clone(nn.Module):
    def forward(self, input, num):
        outputs = []
        for _ in range(num):
            outputs.append(input)
        return outputs
    
class IndexSelect(nn.Module):
    def forward(self, inputs, dim, indices):
        return torch.index_select(inputs, dim, indices)

class einsum(nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)
    
class PositiveAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand((in_dim, out_dim)))
    
    def forward(self, x):
        return torch.matmul(x, torch.exp(self.weight))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.matmul1 = einsum("bhid,bhjd->bhij")
        self.matmul2 = einsum("bhij,bhjd->bhid")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=h)

        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        out = self.matmul2([attn, v])
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x2, self.mlp(self.norm2(x2))])
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(50, 14), patch_size=5, in_chans=3, embed_dim=32):
        super().__init__()
        self.img_size = img_size
        patch_size = (patch_size, 1)
        self.patch_size = patch_size
        self.num_patchs = (img_size[0] // patch_size[0]) * (img_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_size[0], "height:{} != H:{}".format(self.img_size[0], H)
        assert W == self.img_size[1], "width:{} != W:{}".format(self.img_size[1], W)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(50, 14), patch_size=5, in_chans=3, num_classes=6, embed_dim=32, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, mlp_head=False, pre_drop_rate=0., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.pre_dropout = nn.Dropout(pre_drop_rate)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patchs = self.patch_embed.num_patchs

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patchs + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}
    
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])
        x = self.pre_dropout(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x