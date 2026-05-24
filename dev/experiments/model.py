"""
Transformer-based candidate selector.
Sequence Transformer → Cross-Attention with candidates → logit per candidate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT, SOFT_TEMP

SEQ_DIM  = 11
CAND_DIM = 11   # +1: candidate family type feature (base/acc/frenet/turn/jerk/latency)


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

        # Candidate self-attention: candidates compare each other
        # → 후보끼리 상대 비교 가능 → oracle이 다른 후보들 대비 어떤 위치인지 파악
        self.cand_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cand_sa_norm = nn.LayerNorm(d_model)

        # Head: [seq_cls || cand_ctx || cand_feat_raw] → 1 logit per candidate
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + CAND_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_feat:  torch.Tensor,   # (B, 11, SEQ_DIM=11)
        cand_feat: torch.Tensor,   # (B, C, CAND_DIM=11)
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

        # Candidate self-attention: each candidate attends to all other candidates
        # 각 후보가 전체 후보 집합을 보며 상대적 위치 파악 → 상대 랭킹 가능
        sa_ctx, _ = self.cand_self_attn(ctx, ctx, ctx)   # (B, C, d_model)
        ctx       = self.cand_sa_norm(ctx + sa_ctx)       # residual

        # Head
        cls_expand = cls.unsqueeze(1).expand(-1, ctx.size(1), -1)  # (B, C, d_model)
        h = torch.cat([cls_expand, ctx, cand_feat], dim=-1)        # (B, C, d_model*2+CAND_DIM)
        logits = self.head(h).squeeze(-1)                           # (B, C)

        return logits


class TransformerRegressor(nn.Module):
    """
    C-group 전용 직접 회귀 모델.
    CandidateSelector와 상보적: selector가 확신 없을 때(high entropy) 이 모델 우선 사용.

    학습: 전체 10K 샘플, L2 거리 최소화 (= E[dist] 최소화 → R-Hit@1cm 간접 최적화)
    추론: entropy-weighted blend  → pred = (1-α)·selector + α·regressor
          α = normalized_entropy (high entropy → more regressor)
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Sequence encoder (CandidateSelector와 동일 구조)
        self.seq_proj = nn.Linear(SEQ_DIM, d_model)
        self.pos_enc  = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 직접 위치 예측 head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, seq_feat: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """
        seq_feat: (B, 11, SEQ_DIM) — 정규화된 피처
        p0:       (B, 3)           — 마지막 알려진 절대 좌표 (coords[:, -1, :])
        → pred:   (B, 3)           — p0 + learned offset
        """
        seq_h = self.seq_proj(seq_feat)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)
        cls   = seq_h[:, -1, :]   # 마지막 시점 summary
        return p0 + self.head(cls)   # offset 예측 → 절대 좌표 복원


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
    temp: float = 1.0,
) -> torch.Tensor:          # (B, 3)
    """Weighted sum of top-k candidates."""
    weights = F.softmax(logits / temp, dim=-1)      # (B, C)
    topk_w, topk_idx = weights.topk(topk, dim=-1)  # (B, k)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_cands = cands.gather(
        1, topk_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, k, 3)
    return (topk_cands * topk_w.unsqueeze(-1)).sum(dim=1)  # (B, 3)
