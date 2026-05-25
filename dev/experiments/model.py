"""
Transformer-based candidate selector.
Sequence Transformer → Cross-Attention with candidates → logit per candidate

Phase 10 구조 변경:
- SA (Candidate Self-Attention) 제거: oracle rank 악화 원인 확인 (Phase 8)
- Family classifier 추가: CLS → 6개 계열 중 어느 family인지 예측
- Family boost: 각 후보의 계열 점수를 logit에 더함 → 맞는 계열 후보에 집중
- Loss = BCE(candidate) + PW + LML + CE(family)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT, SOFT_TEMP
from candidates import CANDIDATE_FAMILY, FAMILY_NAMES

SEQ_DIM    = 11
CAND_DIM   = 11
N_FAMILIES = len(FAMILY_NAMES)   # 6

# Cached tensor of candidate family indices
_CAND_FAM_CACHE: dict = {}

def _cand_fam_tensor(device):
    key = str(device)
    if key not in _CAND_FAM_CACHE:
        _CAND_FAM_CACHE[key] = torch.tensor(
            CANDIDATE_FAMILY, dtype=torch.long, device=device
        )
    return _CAND_FAM_CACHE[key]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 11):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos).unsqueeze(0)


class CandidateSelector(nn.Module):
    """
    2-stage selector:
      Stage 1 — Family classifier: CLS → which family (6 classes)?
      Stage 2 — Candidate ranker:  candidates × seq → boosted logit per candidate

    forward() returns:
      logits        (B, C) — family-boosted candidate scores (for prediction & BCE/LML/PW losses)
      family_logits (B, 6) — family CE loss target (returned only when return_family=True)
    """
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

        # Stage 1 — Family classifier (CLS token → 6 families)
        self.family_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_FAMILIES),
        )

        # Stage 2 — Candidate scoring head
        # input: [seq_cls || cand_ctx || cand_feat_raw]
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + CAND_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_feat:      torch.Tensor,   # (B, 11, SEQ_DIM)
        cand_feat:     torch.Tensor,   # (B, C, CAND_DIM)
        return_family: bool = False,
    ):
        B = seq_feat.size(0)

        # ── Sequence encoding ───────────────────────────────────────────────
        seq_h = self.seq_proj(seq_feat)   # (B, 11, d)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)   # (B, 11, d)
        cls   = seq_h[:, -1, :]           # (B, d)  마지막 시점 summary

        # ── Stage 1: Family prediction ───────────────────────────────────────
        family_logits = self.family_head(cls)                        # (B, 6)
        family_log_p  = F.log_softmax(family_logits, dim=-1)        # (B, 6)

        # ── Stage 2: Candidate scoring ───────────────────────────────────────
        cand_h = self.cand_proj(cand_feat)                           # (B, C, d)

        # Cross-attention: each candidate attends to full sequence
        ctx, _ = self.cross_attn(cand_h, seq_h, seq_h)              # (B, C, d)
        ctx    = self.cross_norm(cand_h + ctx)                       # residual

        # Head
        cls_exp = cls.unsqueeze(1).expand(-1, ctx.size(1), -1)      # (B, C, d)
        h       = torch.cat([cls_exp, ctx, cand_feat], dim=-1)      # (B, C, d*2+CAND_DIM)
        logits  = self.head(h).squeeze(-1)                           # (B, C)

        # Family boost: add log-prob of each candidate's family to its logit
        # → 모델이 "turn 계열"이라 판단하면 turn 후보들의 logit이 집단적으로 상승
        cand_fam    = _cand_fam_tensor(seq_feat.device)             # (C,)
        fam_boost   = family_log_p[:, cand_fam]                     # (B, C)
        logits      = logits + fam_boost

        if return_family:
            return logits, family_logits
        return logits


class TransformerRegressor(nn.Module):
    """C-group 전용 직접 회귀 모델 (보조)."""
    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(SEQ_DIM, d_model)
        self.pos_enc  = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def forward(self, seq_feat: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        seq_h = self.seq_proj(seq_feat)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)
        cls   = seq_h[:, -1, :]
        return p0 + self.head(cls)


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
