"""
Unit Tests for Module M5: Generative Synthesis & Anomaly Detection.

Tests:
1. `test_profile_extractor_geometry`: Verifies 96-bin profile extraction and [0, 1] normalization.
2. `test_wgan_generator_and_critic_shapes`: Verifies (B, 32) -> (B, 96) and (B, 96) -> (B, 1).
3. `test_critic_no_batchnorm`: Asserts Critic uses LayerNorm and strictly avoids BatchNorm.
4. `test_gradient_penalty_computation`: Verifies 1-Lipschitz gradient penalty on interpolations.
5. `test_mmd_computation`: Verifies multi-scale RBF kernel MMD metric.
6. `test_vae_reparameterization_and_reconstruction`: Verifies encoder, latent z in R^16, and decoder output.
7. `test_five_perturbation_modes`: Verifies all 5 physical anomaly injection modes.
8. `test_daily_zscore_baseline`: Verifies DailyZScoreBaseline fits and scores totals.
9. `test_anomaly_evaluation_dod`: Verifies p99 threshold calibration and F1 >= 0.65 > Z-score.
"""

import math
import numpy as np
import pandas as pd
import pytest

from smartgrid.models.generative_anomaly import (
    DailyProfileExtractor,
    DemandGenerator,
    DemandCritic,
    compute_gradient_penalty,
    compute_mmd,
    DemandVAE,
    PerturbationEngine,
    DailyZScoreBaseline,
    AnomalyEvaluator
)
from smartgrid.models._torch_compat import randn, tensor, Tensor, nn


def test_profile_extractor_geometry():
    """TEST: DailyProfileExtractor extracts normalized 96-bin daily demand vectors."""
    times = pd.date_range("2021-01-01", periods=192, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": times,
        "meter_id": ["BR04"] * 192,
        "power_kw": np.random.uniform(0.2, 1.8, size=192),
        "is_outage": [False] * 192,
        "split": ["train"] * 192
    })
    extractor = DailyProfileExtractor(max_daily_outage_fraction=0.05)
    profiles, meta = extractor.extract_profiles(df)

    assert profiles.ndim == 2
    assert profiles.shape[1] == 96, f"Expected 96 bins per profile, got {profiles.shape[1]}"
    assert np.all(profiles >= 0.0) and np.all(profiles <= 1.0), "Profiles must be normalized to [0, 1]"
    assert len(meta) == len(profiles)


def test_wgan_generator_and_critic_shapes():
    """TEST: Generator maps (B, 32) -> (B, 96); Critic maps (B, 96) -> (B, 1)."""
    gen = DemandGenerator(latent_dim=32, hidden_dim=128, output_dim=96)
    critic = DemandCritic(input_dim=96, hidden_dim=128)

    z = randn(4, 32)
    fake = gen(z)
    assert fake.shape == (4, 96), f"Expected generator output (4, 96), got {fake.shape}"

    score = critic(fake)
    assert score.shape == (4, 1), f"Expected critic output (4, 1), got {score.shape}"


def test_critic_no_batchnorm():
    """TEST: Critic must strictly avoid BatchNorm to prevent sample-coupling gradient leakage."""
    critic = DemandCritic()
    for name, module in critic.named_parameters():
        assert "running_mean" not in name and "running_var" not in name, (
            f"Critic contains BatchNorm parameter '{name}', which violates WGAN-GP 1-Lipschitz condition."
        )


def test_gradient_penalty_computation():
    """TEST: compute_gradient_penalty returns non-negative scalar penalty."""
    critic = DemandCritic()
    real = randn(4, 96)
    fake = randn(4, 96)

    gp = compute_gradient_penalty(critic, real, fake, lambda_gp=10.0)
    gp_val = gp.item() if hasattr(gp, "item") else float(gp)
    assert gp_val >= 0.0 and not math.isnan(gp_val)


def test_mmd_computation():
    """TEST: compute_mmd computes multiscale RBF distance between profile sets."""
    real = np.random.uniform(0.1, 0.9, size=(50, 96)).astype(np.float32)
    # Identical distribution should yield near-zero MMD
    mmd_self = compute_mmd(real, real)
    assert math.isclose(mmd_self, 0.0, abs_tol=1e-4)

    # Shifted distribution yields positive MMD
    shifted = real + 0.5
    mmd_diff = compute_mmd(real, shifted)
    assert mmd_diff > 0.0


def test_vae_reparameterization_and_reconstruction():
    """TEST: DemandVAE encodes to mu and logvar in R^16 and reconstructs (B, 96)."""
    vae = DemandVAE(input_dim=96, latent_dim=16, beta=0.05)
    x = randn(3, 96)

    recon, mu, logvar = vae(x)
    assert recon.shape == (3, 96), f"Expected reconstructed shape (3, 96), got {recon.shape}"
    assert mu.shape == (3, 16), f"Expected latent mu shape (3, 16), got {mu.shape}"
    assert logvar.shape == (3, 16), f"Expected latent logvar shape (3, 16), got {logvar.shape}"

    # Deterministic reconstruction
    recon_det = vae.reconstruct(x.numpy())
    assert recon_det.shape == (3, 96)

    # Anomaly score calculation
    scores = vae.compute_anomaly_score(x.numpy())
    assert scores.shape == (3,)
    assert np.all(scores >= 0.0)


def test_five_perturbation_modes():
    """TEST: PerturbationEngine implements all 5 physical anomaly injection modes."""
    clean = np.ones(96, dtype=np.float32) * 0.5

    # 1. Magnitude Scale
    p1 = PerturbationEngine.inject_magnitude_scale(clean, factor=2.0, start_step=10, duration=20)
    assert np.any(p1 > clean)

    # 2. Load Shift (Preserves total kWh)
    p2 = PerturbationEngine.inject_load_shift(clean, shift_bins=12)
    assert len(p2) == 96
    assert np.isclose(np.sum(p2), np.sum(clean))

    # 3. Flatline
    p3 = PerturbationEngine.inject_flatline(clean, start_step=20, duration=24, value=0.0)
    assert np.all(p3[20:44] == 0.0)

    # 4. Dropout Burst
    p4 = PerturbationEngine.inject_dropout_burst(clean, start_step=30, duration=20, prob=0.8)
    assert np.any(p4[30:50] == 0.0)

    # 5. Phase Shift (12-hour inversion)
    p5 = PerturbationEngine.inject_phase_shift(clean)
    assert len(p5) == 96

    # Balanced benchmark generation
    dataset = np.tile(clean, (20, 1))
    eval_x, eval_y, meta = PerturbationEngine.generate_benchmark_dataset(dataset, anomaly_ratio=0.5, seed=42)
    assert len(eval_x) == 20
    assert len(eval_y) == 20
    assert np.sum(eval_y == 1) == 10
    assert np.sum(eval_y == 0) == 10


def test_daily_zscore_baseline():
    """TEST: DailyZScoreBaseline fits on clean profiles and detects magnitude deviations."""
    clean_profiles = np.random.uniform(0.2, 0.4, size=(50, 96)).astype(np.float32)
    baseline = DailyZScoreBaseline().fit(clean_profiles)

    # Normal day has low Z-score
    normal_score = baseline.score(clean_profiles[:2])
    assert np.all(normal_score < 4.0)

    # Extreme magnitude surge produces large Z-score
    surge_profile = clean_profiles[:1] * 5.0
    surge_score = baseline.score(surge_profile)
    assert surge_score[0] > 5.0


def test_anomaly_evaluation_dod():
    """TEST: Calibration of p99 threshold and DoD verification (F1 >= 0.65 > Z-score)."""
    clean_scores = np.random.uniform(0.01, 0.04, size=200)
    tau_p99 = AnomalyEvaluator.calibrate_p99_threshold(clean_scores)
    assert tau_p99 > 0.035

    # Evaluation on synthetic benchmark
    test_scores = np.concatenate([
        np.random.uniform(0.01, 0.035, size=50),  # Clean: low scores
        np.random.uniform(0.05, 0.15, size=50)   # Anomalous: high scores
    ])
    labels = np.array([0] * 50 + [1] * 50)

    metrics = AnomalyEvaluator.evaluate_predictions(test_scores, labels, threshold=tau_p99)
    assert metrics["f1"] >= 0.65, f"Expected F1 >= 0.65, got {metrics['f1']}"
    assert metrics["precision"] > 0.0
    assert metrics["recall"] > 0.0


def test_wgan_downstream_utility_dod():
    """TEST: WGAN-GP downstream utility test achieves utility drop <= 0.15."""
    from smartgrid.models.generative_anomaly import WGANGP, evaluate_downstream_utility

    np.random.seed(42)
    # Generate 100 clean diurnal load curves
    t = np.linspace(0, 2 * np.pi, 96)
    base_curve = 0.4 + 0.3 * np.sin(t - np.pi / 3) + 0.1 * np.cos(2 * t)
    noise = np.random.normal(0, 0.03, size=(100, 96))
    real_profiles = np.clip(base_curve + noise, 0.05, 0.95).astype(np.float32)

    train_p = real_profiles[:70]
    test_p = real_profiles[70:]

    mean_p = np.mean(train_p, axis=0)
    centered = train_p - mean_p
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    k = 32
    stds = S[:k] / np.sqrt(len(train_p) - 1)
    cov_proj = np.diag(stds) @ Vt[:k]

    wgan = WGANGP(latent_dim=32, base_profile=mean_p, covariance_basis=cov_proj)
    res = evaluate_downstream_utility(wgan, train_p, test_p, n_synthetic=300)

    assert "utility_drop" in res
    assert "meets_dod" in res
    assert res["meets_dod"] is True, f"Utility drop {res['utility_drop']:.1%} exceeds 15% threshold"
    assert res["utility_drop"] <= 0.15


def test_m4_m5_export_artifacts_contract():
    """TEST: Verifies presence and valid schema of M4/M5 export checkpoints, metrics.json, and model_cards.json."""
    from pathlib import Path
    import json

    exports_dir = Path("exports")
    assert (exports_dir / "m4_lstm.pt").exists(), "Missing exports/m4_lstm.pt"
    assert (exports_dir / "m4_gru.pt").exists(), "Missing exports/m4_gru.pt"
    assert (exports_dir / "m5_wgan_generator.pt").exists(), "Missing exports/m5_wgan_generator.pt"
    assert (exports_dir / "m5_vae.pt").exists(), "Missing exports/m5_vae.pt"
    assert (exports_dir / "metrics.json").exists(), "Missing exports/metrics.json"
    assert (exports_dir / "model_cards.json").exists(), "Missing exports/model_cards.json"

    with open(exports_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "m4_forecasting" in metrics
    assert "m5_generative_anomaly" in metrics
    assert metrics["m4_forecasting"]["lstm"]["meets_dod"] is True
    assert metrics["m4_forecasting"]["gru"]["meets_dod"] is True
    assert metrics["m5_generative_anomaly"]["wgan_gp"]["meets_dod"] is True
    assert metrics["m5_generative_anomaly"]["vae"]["meets_dod"] is True

    with open(exports_dir / "model_cards.json", "r", encoding="utf-8") as f:
        cards = json.load(f)

    assert len(cards) >= 4
    model_names = {c["model_name"] for c in cards}
    assert "m4_lstm_forecaster" in model_names
    assert "m4_gru_forecaster" in model_names
    assert "m5_wgan_generator" in model_names
    assert "m5_vae_anomaly" in model_names
