import torch
import torch.nn as nn
from config import D_MODEL, NHEAD, NUM_LAYERS, DROPOUT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        times = torch.arange(11, dtype=torch.float32)
        self.register_buffer("times", times)
        self.linear = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.linear(self.times.unsqueeze(-1))  # (11, d_model)
        return x + pe.unsqueeze(0)


class PhysicsCorrector(nn.Module):
    """
    Transformer that predicts correction to physics baseline.
    Final prediction = physics_pred + correction
    """
    def __init__(
        self,
        in_features: int = 9,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, x: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, 11, 9)
        physics: (B, 3)
        returns: (B, 3) absolute prediction
        """
        h = self.input_proj(x)       # (B, 11, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)      # (B, 11, d_model)
        correction = self.head(h[:, -1, :])  # (B, 3)
        return physics + correction
