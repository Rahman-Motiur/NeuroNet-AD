
import torch
import torch.nn as nn


class MGCA(nn.Module):

    def __init__(self, img_channels: int, embed_dim: int, num_heads: int = 4, token_mode: str = "channels"):
        super().__init__()
        assert token_mode in ("channels", "spatial")
        self.token_mode = token_mode
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project image tokens -> E ; project text tokens -> E
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # For mapping image to embed space depending on tokenization
        if self.token_mode == "channels":
            # We'll consider C tokens; map each channel vector (H*W) -> E using a small MLP
            self.img_tokenizer = nn.Sequential(
                nn.Linear(img_channels, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
            self.img_detoken = nn.Sequential(
                nn.Linear(embed_dim, img_channels)
            )
        else:
            # spatial tokens: flatten H*W tokens, each with C feats -> E
            self.img_tokenizer = nn.Sequential(
                nn.Linear(img_channels, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            )
            self.img_detoken = nn.Sequential(
                nn.Linear(embed_dim, img_channels)
            )

    def forward(self, img: torch.Tensor, txt_tokens: torch.Tensor) -> torch.Tensor:
        # img: [B, C, H, W], txt_tokens: [B, L, E]
        B, C, H, W = img.shape
        if self.token_mode == "channels":
            # pool spatially -> [B, C]
            img_pool = img.mean(dim=(2,3))  # [B, C]
            q_tokens = self.img_tokenizer(img_pool).unsqueeze(1)  # [B, 1, E]; single token represents image summary
        else:
            # spatial tokens: [B, H*W, C] -> project to E
            img_sp = img.permute(0,2,3,1).reshape(B, H*W, C)
            q_tokens = self.img_tokenizer(img_sp)  # [B, H*W, E]

        # Prepare K,V from text/meta tokens
        k = self.k_proj(txt_tokens)  # [B, L, E]
        v = self.v_proj(txt_tokens)  # [B, L, E]
        q = self.q_proj(q_tokens)    # [B, Tq, E]

        attn_out, _ = self.attn(q, k, v)  # [B, Tq, E]
        fused = self.out_proj(attn_out)   # [B, Tq, E]

        if self.token_mode == "channels":
            # Map back to channel summary, then broadcast residually
            ch_delta = self.img_detoken(fused.squeeze(1))  # [B, C]
            ch_delta = ch_delta.view(B, C, 1, 1)
            out = img + img * ch_delta  # residual gating
        else:
            sp = self.img_detoken(fused)  # [B, H*W, C]
            sp = sp.view(B, H, W, C).permute(0,3,1,2)
            out = img + sp
        return out

