"""
Transformer-based candidate selector.
Sequence Transformer → Cross-Attention with candidates → logit per candidate

Phase 5 회귀 구조 (commit 66620d4 기준):
- Smart 50-cand + LISTMLE_WEIGHT=0.10
- soft-CE + PW×0.25 + LML×0.10
- GCN / Aux RegHead 없음 (Phase 8~14 추가분 제거)
- CAND_DIM=10 (family_id/5 없음 — Phase 8에서 −0.73pp 확인)

Experiment A: CAND_FEAT_INTERACTION=True → CAND_DIM=14
- +obs_acc_perp, +par_match, +perp_match, +jerk_match
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT, SOFT_TEMP, CAND_FEAT_INTERACTION

SEQ_DIM  = 11
CAND_DIM = 10 + (4 if CAND_FEAT_INTERACTION else 0)  # 10 (base) or 14 (with interaction)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 11):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Head: [seq_cls || cand_ctx || cand_feat_raw] → 1 logit per candidate
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + CAND_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_feat:  torch.Tensor,   # (B, 11, SEQ_DIM)
        cand_feat: torch.Tensor,   # (B, C, CAND_DIM)
    ) -> torch.Tensor:             # (B, C) logits

        # Sequence encoding
        seq_h = self.seq_proj(seq_feat)   # (B, 11, d)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)   # (B, 11, d)
        cls   = seq_h[:, -1, :]           # (B, d) — last timestep summary

        # Candidate encoding + cross-attention
        cand_h = self.cand_proj(cand_feat)               # (B, C, d)
        ctx, _ = self.cross_attn(cand_h, seq_h, seq_h)  # (B, C, d)
        ctx    = self.cross_norm(cand_h + ctx)            # residual

        # Scoring head
        cls_exp = cls.unsqueeze(1).expand(-1, ctx.size(1), -1)   # (B, C, d)
        h       = torch.cat([cls_exp, ctx, cand_feat], dim=-1)   # (B, C, d*2+CAND_DIM)
        logits  = self.head(h).squeeze(-1)                        # (B, C)

        return logits


def soft_labels(cands: torch.Tensor, true: torch.Tensor, temp: float = SOFT_TEMP) -> torch.Tensor:
    dist = torch.norm(cands - true.unsqueeze(1), dim=-1)
    return F.softmax(-dist / temp, dim=-1)


def selector_predict(
    logits: torch.Tensor,
    cands:  torch.Tensor,
    topk:   int   = 3,
    temp:   float = 1.0,
) -> torch.Tensor:
    """Weighted sum of top-k candidates."""
    weights = F.softmax(logits / temp, dim=-1)
    topk_w, topk_idx = weights.topk(topk, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_cands = cands.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 3))
    return (topk_cands * topk_w.unsqueeze(-1)).sum(dim=1)
