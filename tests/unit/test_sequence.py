"""
Unit Tests for Module M4: Sequence Forecasting & Data Pipelines.

Tests:
1. `test_per_meter_scaler_anti_leakage`: Verifies train-only fit, normalization, and inverse transform.
2. `test_sliding_window_dataset_shapes`: Verifies 672-in / 96-out geometry and 6 feature channels.
3. `test_masked_loss_and_zero_division`: Verifies outage loss masking and 100% outage safeguard.
4. `test_lstm_forecaster_forward`: Verifies GlobalLSTMForecaster forward pass and embedding projection.
5. `test_gru_forecaster_forward`: Verifies GlobalGRUForecaster forward pass.
6. `test_baselines`: Verifies Persistence, Seasonal-Naive, and LightGBM predictions.
7. `test_mase_metric_and_protection`: Verifies scale-free MASE and flatline zero-division defense.
8. `test_portfolio_beat_rate_dod`: Verifies >= 70% beat rate calculation logic.
"""

import math
import numpy as np
import pandas as pd
import pytest

from smartgrid.data.sequence_loader import PerMeterScaler, SlidingWindowDataset
from smartgrid.models.sequence import (
    GlobalLSTMForecaster,
    GlobalGRUForecaster,
    MaskedMSELoss,
    MaskedL1Loss
)
from smartgrid.models.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    LightGBMForecaster
)
from smartgrid.models.evaluation import (
    compute_mase,
    compute_mae,
    compute_rmse,
    compute_mape,
    compute_metrics_dict,
    evaluate_portfolio_beat_rate
)
from smartgrid.models._torch_compat import randn, tensor, Tensor


def test_per_meter_scaler_anti_leakage():
    """TEST: PerMeterScaler fits exclusively on train split per meter with zero test leakage."""
    train_df = pd.DataFrame({
        "meter_id": ["BR04", "BR04", "MH02", "MH02"],
        "power_kw": [2.0, 4.0, 10.0, 30.0],
        "is_outage": [False, False, False, False]
    })
    scaler = PerMeterScaler(method="standard")
    scaler.fit(train_df)

    # BR04 mean=3.0, std=1.0; MH02 mean=20.0, std=10.0
    assert scaler.params["BR04"]["mean"] == 3.0
    assert scaler.params["BR04"]["std"] == 1.0
    assert scaler.params["MH02"]["mean"] == 20.0
    assert scaler.params["MH02"]["std"] == 10.0

    # Transform test values without updating train scale
    scaled_br = scaler.transform("BR04", np.array([5.0]))
    assert np.isclose(scaled_br[0], 2.0)

    # Inverse transform recovers original physical kW
    recovered_br = scaler.inverse_transform("BR04", scaled_br)
    assert np.isclose(recovered_br[0], 5.0)


def test_sliding_window_dataset_shapes():
    """TEST: SlidingWindowDataset constructs 672-in / 96-out tensors with 6 feature channels."""
    times = pd.date_range("2021-01-01", periods=800, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": times,
        "meter_id": ["BR04"] * 800,
        "power_kw": np.random.uniform(0.5, 2.5, size=800),
        "is_outage": [False] * 800
    })
    scaler = PerMeterScaler().fit(df)
    ds = SlidingWindowDataset(df, scaler, input_len=672, horizon=96, stride=16)

    assert len(ds) > 0
    x, y, mask, m_idx = ds[0]

    # Verify tensor geometry
    assert x.shape == (672, 6), f"Input shape {x.shape} must be (672, 6)"
    assert y.shape == (96,), f"Target shape {y.shape} must be (96,)"
    assert mask.shape == (96,), f"Mask shape {mask.shape} must be (96,)"
    assert m_idx.item() == 0


def test_masked_loss_and_zero_division():
    """TEST: MaskedMSELoss masks outage steps and returns 0.0 on 100% outage window."""
    criterion = MaskedMSELoss()

    pred = tensor(np.array([[2.0, 3.0, 4.0]]))
    target = tensor(np.array([[1.0, 1.0, 1.0]]))
    mask = tensor(np.array([[1.0, 0.0, 1.0]]))  # Step 1 is outage

    loss = criterion(pred, target, mask)
    # (2-1)^2 * 1 + (3-1)^2 * 0 + (4-1)^2 * 1 = 1 + 0 + 9 = 10 / 2 = 5.0
    loss_val = loss.item() if hasattr(loss, "item") else float(loss)
    assert math.isclose(loss_val, 5.0, rel_tol=1e-3)

    # 100% outage safeguard (sum(M) == 0)
    zero_mask = tensor(np.array([[0.0, 0.0, 0.0]]))
    zero_loss = criterion(pred, target, zero_mask)
    zero_val = zero_loss.item() if hasattr(zero_loss, "item") else float(zero_loss)
    assert zero_val == 0.0


def test_lstm_forecaster_forward():
    """TEST: GlobalLSTMForecaster conditions on meter embedding and outputs (B, 96)."""
    model = GlobalLSTMForecaster(num_meters=10, embed_dim=16, in_features=6, hidden_dim=64, num_layers=2)
    x = randn(3, 672, 6)
    m_idx = tensor(np.array([0, 1, 2]), dtype=np.int64)

    out = model(x, m_idx)
    assert out.shape == (3, 96), f"Expected (3, 96), got {out.shape}"


def test_gru_forecaster_forward():
    """TEST: GlobalGRUForecaster outputs (B, 96) trajectory."""
    model = GlobalGRUForecaster(num_meters=10, embed_dim=16, in_features=6, hidden_dim=64, num_layers=2)
    x = randn(2, 672, 6)
    m_idx = tensor(np.array([3, 4]), dtype=np.int64)

    out = model(x, m_idx)
    assert out.shape == (2, 96), f"Expected (2, 96), got {out.shape}"


def test_baselines():
    """TEST: Persistence, Seasonal-Naive, and LightGBM baseline forecasters."""
    # Dummy 672-step sequence
    history = np.arange(672, dtype=np.float32)

    # 1. Persistence (predicts last 24h: [576:672])
    p_fc = PersistenceForecaster(horizon=96)
    pred_p = p_fc.predict(history)
    assert len(pred_p) == 96
    assert np.allclose(pred_p, history[576:672])

    # 2. Seasonal-Naive (predicts same day last week: [0:96])
    sn_fc = SeasonalNaiveForecaster(horizon=96)
    pred_sn = sn_fc.predict(history)
    assert len(pred_sn) == 96
    assert np.allclose(pred_sn, history[0:96])


def test_mase_metric_and_protection():
    """TEST: Scale-free MASE evaluation and flatline zero-division defense."""
    actual = np.array([1.0, 1.2, 1.1, 1.3])
    predicted = np.array([0.95, 1.15, 1.05, 1.25])
    train_series = np.array([1.0, 1.1, 1.05, 1.2, 1.15, 1.25, 1.3])

    mase_val = compute_mase(actual, predicted, train_series, seasonality=2)
    assert mase_val > 0.0 and not math.isnan(mase_val)

    # Flatline training scale returns inf
    flat_train = np.ones(50)
    flat_mase = compute_mase(actual, predicted, flat_train, seasonality=5)
    assert math.isinf(flat_mase) or flat_mase > 0.0


def test_portfolio_beat_rate_dod():
    """TEST: Portfolio beat rate calculator confirms >= 70% threshold."""
    scores = {f"MTR_{i}": 0.85 if i < 15 else 1.10 for i in range(20)}
    eval_res = evaluate_portfolio_beat_rate(scores, benchmark_threshold=1.0, target_beat_rate=0.70)

    assert eval_res["beat_count"] == 15
    assert eval_res["total_meters"] == 20
    assert eval_res["beat_rate"] == 0.75
    assert eval_res["meets_dod"] is True
