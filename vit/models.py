import torch
from torch import nn
from .vit import PatchEmbed, PositionEmbed, Adapter, PositiveAdapter, Mlp

class SingleDecoder(nn.Module):
    def __init__(
            self, input_dim, output_dim, dim=128, length=100, patch_size=(5, 1), in_chans=1, num_heads=8, depth=6,
            adapter_dim=16, adapter_drop=0., tr_dropout=0., head_drop=0., positive_adapter=False, mlp_head=False, mlp_rate=4.
        ):
        super().__init__()
        if positive_adapter:
            self.adapter = PositiveAdapter(in_dim=input_dim, out_dim=adapter_dim, drop=adapter_drop)
        else:
            self.adapter = Adapter(in_dim=input_dim, out_dim=adapter_dim, bias=False, drop=adapter_drop)
        self.patch_embed = PatchEmbed(dim=dim, patch_size=patch_size, in_chans=in_chans)
        ph, pw = patch_size
        self.pos_embed = PositionEmbed(dim=dim, n_patches=(length//ph) * (adapter_dim//pw))
        self.tr = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dropout=tr_dropout, batch_first=True
            ),
            num_layers=depth
        )
        if mlp_head:
            self.head = Mlp(in_dim=dim, hidden_dim=int(dim*mlp_rate), out_dim=output_dim, drop=head_drop)
        else:
            self.head = Adapter(in_dim=dim, out_dim=output_dim, drop=head_drop)
        self.soft_max = nn.Softmax(dim=-1)
        self.img_size = (length, adapter_dim)
        self.patch_size = patch_size

    def forward(self, x) -> torch.Tensor:
        x = self.adapter(x)
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.tr(x)
        return self.head(x)[:, 0, :]
    
    def pred(self, x) -> torch.Tensor:
        x = self.forward(x)
        return self.soft_max(x)