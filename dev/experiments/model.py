"""
Transformer-based candidate selector.
Sequence Transformer → Cross-Attention with candidates → logit per candidate

Phase 11 Step 2 구조:
- soft-CE + PW + LML (Phase 7 proven best)
- CAND_DIM=10 (family_id/5 제거 — Phase 8에서 −0.73pp 확인)
- Auxiliary Regression Head: CLS → rough position Δ
  - CLS가 "어디로 갈지"를 인코딩하도록 강제
  - cross-attention이 올바른 방향 후보에 집중 → oracle rank 개선
  - 학습 시 return_reg=True → rough_delta 반환
  - 추론 시 return_reg=False → logits만 반환 (기존 코드 호환)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT, SOFT_TEMP, EPS

SEQ_DIM  = 11
CAND_DIM = 10  # family_id/5 제거 (Phase 8 추가됐으나 −0.73pp → 제거)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 11):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos).unsqueeze(0)


class CandidateSelector(nn.Module):
    """
    Phase 11 Step 2:
      - Sequence Transformer → CLS
      - Auxiliary reg_head: CLS → rough_Δ (3D offset from p0)
        → loss_reg = smooth_l1( rough_pred / threshold, true / threshold )
        → CLS가 위치 정보를 인코딩하도록 강제 → cross-attention 품질 향상
      - Cross-Attention: candidates ← seq → candidate scores
      - Loss = soft-CE + PW + LML + REG_WEIGHT × loss_reg
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

        # Auxiliary regression head: CLS → rough 3D displacement from p0
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

        # Candidate encoder
        self.cand_proj = nn.Linear(CAND_DIM, d_model)

        # Cross-attention: candidate queries, sequence keys/values
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Scoring head: [seq_cls || cand_ctx || cand_feat_raw] → 1 logit per candidate
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + CAND_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_feat:   torch.Tensor,        # (B, 11, SEQ_DIM)
        cand_feat:  torch.Tensor,        # (B, C, CAND_DIM)
        return_reg: bool = False,
    ):
        # ── Sequence encoding ────────────────────────────────────────────────
        seq_h = self.seq_proj(seq_feat)  # (B, 11, d)
        seq_h = self.pos_enc(seq_h)
        seq_h = self.seq_encoder(seq_h)  # (B, 11, d)
        cls   = seq_h[:, -1, :]          # (B, d) — 마지막 시점 summary

        # ── Auxiliary regression (학습 전용) ─────────────────────────────────
        rough_delta = self.reg_head(cls)  # (B, 3) — rough Δ from p0

        # ── Candidate scoring ────────────────────────────────────────────────
        cand_h = self.cand_proj(cand_feat)                 # (B, C, d)
        ctx, _ = self.cross_attn(cand_h, seq_h, seq_h)    # (B, C, d)
        ctx    = self.cross_norm(cand_h + ctx)             # residual

        cls_exp = cls.unsqueeze(1).expand(-1, ctx.size(1), -1)   # (B, C, d)
        h       = torch.cat([cls_exp, ctx, cand_feat], dim=-1)   # (B, C, d*2+CAND_DIM)
        logits  = self.head(h).squeeze(-1)                        # (B, C)

        if return_reg:
            return logits, rough_delta
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


class MosquitoLSTM(nn.Module):
    """
    BiLSTM direct regression for mosquito position.
    Frenet-parametric output: pred = p0 + d1_s·d1 + par·acc_par + perp·acc_perp + jerk_s·jerk
    This output is yaw-invariant because seq_features are rotation-invariant and the
    Frenet coefficients map to XYZ via the actual trajectory's tangent/normal frame.

    Trained with C-group 10× weighted smooth_l1 loss to focus on borderline cases
    where the selector's 52 physical candidates all miss (nearest candidate > 1cm).
    """
    def __init__(self, hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=SEQ_DIM,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, 4)   # (d1_scale, par, perp, jerk_scale)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[0] = 2.0           # init: pure linear extrapolation

    def forward(self, seq_feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        seq_feat : (B, 11, SEQ_DIM) — rotation-invariant sequence features
        coords   : (B, 11, 3)       — raw 3D trajectory
        Returns  : (B, 3)           — predicted next position
        """
        _, (h_n, _) = self.lstm(seq_feat)
        h      = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden*2) last-layer fwd+bwd
        params = self.head(h)                             # (B, 4)

        p0       = coords[:, 10]
        d1       = coords[:, 10] - coords[:, 9]
        d2       = coords[:, 9]  - coords[:, 8]
        acc      = d1 - d2
        prev_acc = d2 - (coords[:, 8] - coords[:, 7])
        jerk     = acc - prev_acc

        speed    = d1.norm(dim=-1, keepdim=True).clamp(min=EPS)
        tangent  = d1 / speed
        acc_par  = (acc * tangent).sum(-1, keepdim=True) * tangent  # (B, 3) tangential
        acc_perp = acc - acc_par                                      # (B, 3) normal

        return (
            p0
            + params[:, 0:1] * d1        # velocity term
            + params[:, 1:2] * acc_par   # parallel acceleration
            + params[:, 2:3] * acc_perp  # perpendicular acceleration
            + params[:, 3:4] * jerk      # jerk correction
        )


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
