"""
Transformer-based candidate selector.
Sequence Transformer → Cross-Attention with candidates → logit per candidate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT, SOFT_TEMP

SEQ_DIM  = 9
CAND_DIM = 10


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 11):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos).unsqueeze(0)


class CandidateSelector(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Sequence encoder
        self.seq_proj = nn.Linear(SEQ_DIM, d_model)
        self.pos_enc  = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Candidate encoder
        self.cand_proj = nn.Linear(CAND_DIM, d_model)

        # Cross-attention: candidate queries, sequence keys/values
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Head: [seq_cls || cand_ctx || cand_feat_proj] → 1 logit per candidate
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + CAND_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_feat:  torch.Tensor,   # (B, 11, 9)
        cand_feat: torch.Tensor,   # (B, C, 10)
    ) -> torch.Tensor:             # (B, C) logits

        B = seq_feat.size(0)

        # Encode sequence
        seq_h = self.seq_proj(seq_feat)      # (B, 11, d_model)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)      # (B, 11, d_model)
        cls   = seq_h[:, -1, :]              # (B, d_model) — 마지막 시점 summary

        # Encode candidates
        cand_h = self.cand_proj(cand_feat)   # (B, C, d_model)

        # Cross-attention: each candidate attends to full sequence
        ctx, _ = self.cross_attn(cand_h, seq_h, seq_h)  # (B, C, d_model)
        ctx    = self.cross_norm(cand_h + ctx)            # residual

        # Head
        cls_expand = cls.unsqueeze(1).expand(-1, ctx.size(1), -1)  # (B, C, d_model)
        h = torch.cat([cls_expand, ctx, cand_feat], dim=-1)        # (B, C, d_model*2+CAND_DIM)
        logits = self.head(h).squeeze(-1)                           # (B, C)

        return logits


def soft_labels(cands: torch.Tensor, true: torch.Tensor, temp: float = SOFT_TEMP) -> torch.Tensor:
    """
    Distance-weighted soft targets over candidates.
    cands: (B, C, 3), true: (B, 3)
    returns: (B, C) soft label distribution
    """
    dist = torch.norm(cands - true.unsqueeze(1), dim=-1)  # (B, C)
    return F.softmax(-dist / temp, dim=-1)


def selector_predict(
    logits: torch.Tensor,   # (B, C)
    cands:  torch.Tensor,   # (B, C, 3)
    topk: int = 3,
) -> torch.Tensor:          # (B, 3)
    """Weighted sum of top-k candidates."""
    weights = F.softmax(logits, dim=-1)             # (B, C)
    topk_w, topk_idx = weights.topk(topk, dim=-1)  # (B, k)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_cands = cands.gather(
        1, topk_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, k, 3)
    return (topk_cands * topk_w.unsqueeze(-1)).sum(dim=1)  # (B, 3)
