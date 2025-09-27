
import torch
import torch.nn as nn
import torchvision.models as models
from .cbam import CBAM
from .mgca import MGCA


class ResLayerWithCBAM(nn.Module):
    """
    Wraps a torchvision ResNet layer (nn.Sequential of 2 BasicBlocks) to insert CBAM
    between block0 and block1 and apply the residual-style combination described.
    """
    def __init__(self, layer_seq: nn.Sequential, channels: int):
        super().__init__()
        assert isinstance(layer_seq, nn.Sequential) and len(layer_seq) >= 2, "Expected a layer with >=2 BasicBlocks"
        self.block0 = layer_seq[0]
        self.block1_and_rest = nn.Sequential(*list(layer_seq.children())[1:])
        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o0 = self.block0(x)
        o0_cbam = self.cbam(o0)
        # Residual fusion before passing to next blocks
        fused = o0 + o0_cbam
        out = self.block1_and_rest(fused)
        return out


class MetaMLPEncoder(nn.Module):
    """
    Encodes numeric metadata into a token sequence [B, L, E].
    If you have text strings, replace this encoder with a BERT-based one.
    """
    def __init__(self, in_dim: int, embed_dim: int, num_tokens: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pos = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        self.token_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, meta_vec: torch.Tensor) -> torch.Tensor:
        # meta_vec: [B, D]
        emb = self.fc(meta_vec)  # [B, E]
        tokens = self.token_proj(emb).unsqueeze(1).repeat(1, self.num_tokens, 1)  # [B, L, E]
        return tokens + self.pos


class NeuroNetAD(nn.Module):
    def __init__(self, num_classes: int = 3, meta_dim: int = 0, img_token_mode: str = "channels",
                 mgca_heads: int = 4, mgca_embed: int = 256):
        super().__init__()
        # ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Keep stem
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # Replace layers with CBAM-wrapped
        self.layer1 = ResLayerWithCBAM(resnet.layer1, channels=64)
        self.layer2 = ResLayerWithCBAM(resnet.layer2, channels=128)
        self.layer3 = ResLayerWithCBAM(resnet.layer3, channels=256)
        self.layer4 = ResLayerWithCBAM(resnet.layer4, channels=512)

        # Meta encoder -> tokens
        self.meta_dim = meta_dim
        if meta_dim and meta_dim > 0:
            self.meta_enc = MetaMLPEncoder(in_dim=meta_dim, embed_dim=mgca_embed, num_tokens=16)
            self.mgca = MGCA(img_channels=256, embed_dim=mgca_embed, num_heads=mgca_heads, token_mode=img_token_mode)
        else:
            self.meta_enc = None
            self.mgca = None

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        # x: [B, 1 or 3, H, W] -> ensure 3 channels for resnet
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)  # [B, 256, H', W']

        if self.mgca is not None and meta is not None and meta.numel() > 0:
            txt_tokens = self.meta_enc(meta)  # [B, L, E]
            f3 = self.mgca(f3, txt_tokens)    # fused

        f4 = self.layer4(f3)                 # [B, 512, H'', W'']
        pooled = self.avgpool(f4).flatten(1)
        out = self.fc(pooled)
        return out

