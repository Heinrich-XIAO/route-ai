import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            p=p,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    """U-Net style decoder with skip connections"""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=768,
        output_channels=3,
        base_channels=64,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Decoder upsampling layers: 16->32->64->128->256 (4 upsamples)
        self.decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, base_channels * 8, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(base_channels * 8),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        base_channels * 8,
                        base_channels * 4,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(base_channels * 4),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        base_channels * 4,
                        base_channels * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.ReLU(inplace=True),
                ),
                nn.ConvTranspose2d(
                    base_channels * 2,
                    output_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x):
        # Remove cls token
        x = x[:, 1:, :]

        # Reshape to image patches
        B = x.shape[0]
        H = W = int(self.n_patches**0.5)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Upsample through decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        return x


class PCBTransformer(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            p=p,
            attn_p=attn_p,
        )

        self.decoder = TransformerDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            output_channels=in_chans,
            base_channels=64,
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
