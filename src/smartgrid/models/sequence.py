"""
SmartgridAI Module M4: Sequence Forecasting Models & Masked Objectives.

Implements:
1. `GlobalLSTMForecaster`: Multi-meter sequence forecaster conditioning on 16-dim
   meter embeddings, 6 input channels, 2 stacked LSTM recurrent layers, and
   direct multi-step linear projection head (672 steps in -> 96 steps out).
2. `GlobalGRUForecaster`: Multi-meter sequence forecaster with GRU recurrent backbone.
3. `MaskedMSELoss` & `MaskedL1Loss`: Loss functions masked strictly at outage intervals
   with robust zero-division safeguards when batches contain 100% outages.
4. `train_sequence_model`: Training harness with AdamW, learning rate scheduling,
   gradient norm clipping (1.0), and deterministic seed control.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np

from smartgrid.models._torch_compat import (
    nn,
    optim,
    tensor,
    Tensor,
    cat,
    clip_grad_norm,
    manual_seed
)


class GlobalLSTMForecaster(nn.Module):
    """
    Direct multi-step day-ahead sequence forecaster with LSTM recurrent backbone.

    Geometry:
    - Input tensor: (B, 672, 6)
      Channel 0: Scaled Active Power
      Channel 1: Outage flag
      Channel 2: Hour-of-day sine
      Channel 3: Hour-of-day cosine
      Channel 4: Day-of-week sine
      Channel 5: Day-of-week cosine
    - Meter tensor: (B,) integer meter tokens in [0, num_meters - 1]
    - Output tensor: (B, 96) future 24-hour demand projection
    """

    def __init__(
        self,
        num_meters: int = 56,
        embed_dim: int = 16,
        in_features: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        horizon: int = 96
    ) -> None:
        super().__init__()
        self.num_meters = num_meters
        self.embed_dim = embed_dim
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        # Categorical meter embedding
        self.meter_embed = nn.Embedding(num_meters, embed_dim)

        # 2-layer LSTM backbone
        # Input size = 6 sequence features + 16-dim meter embedding
        self.lstm = nn.LSTM(
            input_size=in_features + embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Projection head: maps last hidden state + meter embedding to 96 steps
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x: Tensor, meter_idx: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            x: (B, 672, 6) input sequence
            meter_idx: (B,) integer meter tokens
        Returns:
            (B, 96) direct multi-step forecast
        """
        # Embed meter ID: (B, embed_dim)
        emb = self.meter_embed(meter_idx)

        # Expand embedding across sequence dimension: (B, 672, embed_dim)
        b_sz = x.shape[0]
        seq_len = x.shape[1]
        emb_expanded = emb.unsqueeze(1).expand(b_sz, seq_len, self.embed_dim)

        # Concatenate sequence features with meter embedding: (B, 672, 6 + 16 = 22)
        x_conditioned = cat([x, emb_expanded], dim=-1)

        # Recurrent pass
        lstm_out, _ = self.lstm(x_conditioned)

        # Take final time step hidden representation: (B, hidden_dim)
        last_hidden = lstm_out[:, -1, :]

        # Concatenate final hidden state with meter embedding: (B, 64 + 16 = 80)
        head_in = cat([last_hidden, emb], dim=-1)

        # Multi-step projection: (B, 96)
        pred = self.head(head_in)
        return pred


class GlobalGRUForecaster(nn.Module):
    """
    Direct multi-step day-ahead sequence forecaster with GRU recurrent backbone.
    Matches GlobalLSTMForecaster geometry and projection head.
    """

    def __init__(
        self,
        num_meters: int = 56,
        embed_dim: int = 16,
        in_features: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        horizon: int = 96
    ) -> None:
        super().__init__()
        self.num_meters = num_meters
        self.embed_dim = embed_dim
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.meter_embed = nn.Embedding(num_meters, embed_dim)

        self.gru = nn.GRU(
            input_size=in_features + embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x: Tensor, meter_idx: Tensor) -> Tensor:
        emb = self.meter_embed(meter_idx)
        b_sz = x.shape[0]
        seq_len = x.shape[1]
        emb_expanded = emb.unsqueeze(1).expand(b_sz, seq_len, self.embed_dim)

        x_conditioned = cat([x, emb_expanded], dim=-1)
        gru_out, _ = self.gru(x_conditioned)

        last_hidden = gru_out[:, -1, :]
        head_in = cat([last_hidden, emb], dim=-1)
        pred = self.head(head_in)
        return pred


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss computed exclusively over non-outage intervals.

    L = sum(M * (y_pred - y_true)^2) / (sum(M) + eps)
    where M_t = 0 if interval t is in blackout/outage.

    Safeguard: If sum(M) == 0 (batch is 100% outage), returns 0.0 with requires_grad=True.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        diff_sq = (pred - target) ** 2
        masked_diff = diff_sq * mask
        valid_count = mask.sum()

        valid_num = valid_count.item() if hasattr(valid_count, "item") else float(valid_count)
        if valid_num == 0.0:
            return tensor(0.0, requires_grad=True)

        loss = masked_diff.sum() / (valid_count + self.eps)
        return loss


class MaskedL1Loss(nn.Module):
    """
    Mean Absolute Error loss computed exclusively over non-outage intervals.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        diff_abs = Tensor(np.abs(pred._data - target._data) if hasattr(pred, "_data") else (pred - target).abs())
        masked_diff = diff_abs * mask
        valid_count = mask.sum()

        valid_num = valid_count.item() if hasattr(valid_count, "item") else float(valid_count)
        if valid_num == 0.0:
            return tensor(0.0, requires_grad=True)

        loss = masked_diff.sum() / (valid_count + self.eps)
        return loss


def train_sequence_model(
    model: nn.Module,
    train_loader: Any,
    valid_loader: Optional[Any] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    clip_grad: float = 1.0,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Standardized sequence model training loop with AdamW, gradient clipping,
    and loss history logging.
    """
    manual_seed(seed)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MaskedMSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        batches = train_loader.get_batches(batch_size=32, shuffle=True) if hasattr(train_loader, "get_batches") else train_loader

        for batch_x, batch_y, batch_mask, batch_m in batches:
            optimizer.zero_grad()
            pred = model(batch_x, batch_m)
            loss = criterion(pred, batch_y, batch_mask)
            loss.backward()

            if clip_grad > 0:
                clip_grad_norm(model.parameters(), max_norm=clip_grad)

            optimizer.step()
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            train_losses.append(loss_val)

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(mean_train)
        history["lr"].append(float(optimizer.lr if hasattr(optimizer, "lr") else lr))

        mean_valid = None
        if valid_loader is not None:
            model.eval()
            valid_losses = []
            v_batches = valid_loader.get_batches(batch_size=64, shuffle=False) if hasattr(valid_loader, "get_batches") else valid_loader
            for vx, vy, vmask, vm in v_batches:
                vpred = model(vx, vm)
                vloss = criterion(vpred, vy, vmask)
                valid_losses.append(vloss.item() if hasattr(vloss, "item") else float(vloss))
            mean_valid = float(np.mean(valid_losses)) if valid_losses else 0.0
            history["valid_loss"].append(mean_valid)
            scheduler.step(mean_valid)

        if verbose and (epoch == 1 or epoch % 2 == 0 or epoch == epochs):
            v_str = f" | Valid Loss: {mean_valid:.4f}" if mean_valid is not None else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_train:.4f}{v_str}")

    return {
        "model": model,
        "history": history,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_valid_loss": history["valid_loss"][-1] if history["valid_loss"] else None
    }
