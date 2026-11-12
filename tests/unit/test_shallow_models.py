"""
Unit Tests for SmartgridAI Shallow Neural Network Library (Module M1).
Tests:
- 2D Perceptron Toy Lab (Feature 16)
- XOR 2-2-1 MLP Solver (Feature 17)
- 15-Minute Demand Feature Engineering & Outage Masking (Feature 18)
- Demand MLP Regressor (Feature 18)
- Closed-Form Ridge Regression Baseline (Feature 19)
- MLP vs Ridge Comparative Verification Benchmark
- Model Weight Serialization
"""

import math
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smartgrid.models.shallow import (
    generate_linearly_separable_data,
    Perceptron2D,
    XORMLPSolver,
    build_demand_features,
    generate_synthetic_demand_series,
    DemandMLPRegressor,
    RidgeRegressionBaseline,
    compare_mlp_and_ridge
)


# =====================================================================
# Feature 16: 2D Perceptron Tests
# =====================================================================

def test_perceptron_separable_data_generation():
    """Verify synthetic 2D linearly separable dataset properties."""
    X, y = generate_linearly_separable_data(n_samples=100, seed=42, margin=0.6)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(np.unique(y)) == {0, 1}
    assert np.sum(y == 0) == 50
    assert np.sum(y == 1) == 50


def test_perceptron_training_and_convergence():
    """Verify Perceptron achieves 100% training accuracy on linearly separable data."""
    X, y = generate_linearly_separable_data(n_samples=100, seed=42, margin=0.6)
    model = Perceptron2D(learning_rate=0.1, max_epochs=100, seed=42)
    model.fit(X, y)
    
    assert model.converged is True
    assert model.epochs_trained < 100
    assert model.score(X, y) == 1.0
    assert len(model.weights) == 2
    assert isinstance(model.bias, float)


def test_perceptron_decision_boundary_geometry():
    """Verify Perceptron computes analytical decision boundary line."""
    X, y = generate_linearly_separable_data(n_samples=80, seed=101, margin=0.5)
    model = Perceptron2D(learning_rate=0.05, max_epochs=100, seed=101)
    model.fit(X, y)
    
    boundary = model.get_decision_boundary()
    assert "slope" in boundary
    assert "intercept" in boundary
    assert "equation" in boundary
    assert not math.isnan(boundary["slope"])
    assert not math.isnan(boundary["intercept"])


def test_perceptron_invalid_feature_dim_raises_error():
    """Verify Perceptron raises ValueError when given non-2D feature dimensions."""
    model = Perceptron2D()
    X_3d = np.ones((10, 3))
    y = np.ones(10)
    with pytest.raises(ValueError, match="requires exactly 2 features"):
        model.fit(X_3d, y)


# =====================================================================
# Feature 17: XOR 2-2-1 MLP Solver Tests
# =====================================================================

def test_xor_solver_single_run_hidden_convergence():
    """Verify single XOR solver instance converges with hidden layer."""
    solver = XORMLPSolver(hidden=True, learning_rate=0.1, max_epochs=1500, seed=42)
    solver.fit()
    
    assert solver.converged is True
    assert solver.final_loss < 0.01
    
    # Check predictions match XOR truth table
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    preds = solver.predict(X_xor)
    expected = np.array([[0], [1], [1], [0]])
    np.testing.assert_array_equal(preds, expected)


def test_xor_multi_seed_convergence_with_hidden():
    """DoD: Over 20 random seeds, 2-2-1 MLP with hidden layer solves XOR in >= 19/20 seeds."""
    benchmark = XORMLPSolver.benchmark_xor_seeds(n_seeds=20, hidden=True)
    assert benchmark["total_seeds"] == 20
    assert benchmark["convergences"] >= 19, f"Hidden XOR convergences ({benchmark['convergences']}/20) must be >= 19"
    assert benchmark["passed_dod"] is True


def test_xor_multi_seed_failure_without_hidden():
    """DoD: Over 20 random seeds, single-layer perceptron (no hidden layer) solves XOR in <= 2/20 seeds."""
    benchmark = XORMLPSolver.benchmark_xor_seeds(n_seeds=20, hidden=False)
    assert benchmark["total_seeds"] == 20
    assert benchmark["convergences"] <= 2, f"No-hidden XOR convergences ({benchmark['convergences']}/20) must be <= 2"
    assert benchmark["passed_dod"] is True


def test_xor_continuous_decision_surface():
    """Verify continuous decision surface probability evaluation on [-0.5, 1.5]^2."""
    solver = XORMLPSolver(hidden=True, seed=42)
    solver.fit()
    
    surface_data = solver.evaluate_decision_surface(resolution=30, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5))
    assert surface_data["shape"] == [30, 30]
    assert len(surface_data["x"]) == 30
    assert len(surface_data["y"]) == 30
    
    grid = np.array(surface_data["surface"])
    assert grid.shape == (30, 30)
    assert np.all(grid >= 0.0) and np.all(grid <= 1.0)


def test_xor_predict_before_fit_raises_error():
    """Verify calling predict on untrained XOR solver raises RuntimeError."""
    solver = XORMLPSolver()
    with pytest.raises(RuntimeError, match="has not been trained yet"):
        solver.predict(np.array([[0, 1]]))


# =====================================================================
# Feature 18 & 19: Demand Feature Pipeline & Regressors
# =====================================================================

def test_demand_feature_engineering_lags_and_calendar():
    """Verify lag construction (t-1..t-4, t-96, t-672), rolling stats, and calendar cyclical features."""
    syn = generate_synthetic_demand_series(n_steps=1000, seed=42)
    feats = build_demand_features(
        syn["power_kw"],
        voltages=syn["voltage_v"],
        currents=syn["current_a"]
    )
    
    X = feats["X"]
    y = feats["y"]
    mask = feats["mask"]
    
    # 1000 steps - 672 max lag = 328 samples
    assert len(X) == 328
    assert len(y) == 328
    assert len(mask) == 328
    assert X.shape[1] == 12  # 6 lags + 2 rolling + 4 calendar cyclical features


def test_demand_outage_masking_condition():
    """Verify that intervals where voltage <= 5V or current <= 0.01A are masked."""
    n_steps = 800
    series = np.ones(n_steps) * 1.5
    voltages = np.ones(n_steps) * 230.0
    currents = np.ones(n_steps) * 5.0
    
    # Inject outage at index 700
    voltages[700] = 3.0
    currents[700] = 0.0
    
    feats = build_demand_features(series, voltages=voltages, currents=currents)
    # Target index corresponding to step 700 is 700 - 672 = 28
    assert feats["mask"][28] is True or feats["mask"][28] == 1
    # Step 701 is normal
    assert feats["mask"][29] is False or feats["mask"][29] == 0


def test_demand_mlp_fit_predict_evaluate():
    """Verify DemandMLPRegressor training loop, non-negative predictions, and finite metrics."""
    syn = generate_synthetic_demand_series(n_steps=1200, seed=42)
    feats = build_demand_features(syn["power_kw"], voltages=syn["voltage_v"], currents=syn["current_a"])
    X, y, mask = feats["X"], feats["y"], feats["mask"]
    
    split_idx = int(len(X) * 0.7)
    X_train, y_train, mask_train = X[:split_idx], y[:split_idx], mask[:split_idx]
    X_test, y_test, mask_test = X[split_idx:], y[split_idx:], mask[split_idx:]
    
    model = DemandMLPRegressor(input_dim=X.shape[1], seed=42)
    hist = model.fit(X_train, y_train, mask=mask_train, epochs=40, batch_size=32)
    
    assert len(hist["train_loss"]) == 40
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert np.all(preds >= 0.0)
    
    eval_res = model.evaluate(X_test, y_test, mask=mask_test)
    assert eval_res["mae"] > 0.0
    assert eval_res["rmse"] > 0.0
    assert not math.isnan(eval_res["r2"])


def test_ridge_regression_closed_form_solution():
    """Verify RidgeRegressionBaseline closed-form fitting and metric computation."""
    syn = generate_synthetic_demand_series(n_steps=1000, seed=42)
    feats = build_demand_features(syn["power_kw"], voltages=syn["voltage_v"], currents=syn["current_a"])
    X, y, mask = feats["X"], feats["y"], feats["mask"]
    
    ridge = RidgeRegressionBaseline(alpha=1.0)
    ridge.fit(X, y, mask=mask)
    
    assert ridge.weights is not None
    assert len(ridge.weights) == X.shape[1] + 1  # Weights include bias term
    
    preds = ridge.predict(X)
    assert np.all(preds >= 0.0)
    
    metrics = ridge.evaluate(X, y, mask=mask)
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0


def test_demand_mlp_strictly_beats_ridge_mae():
    """DoD: Verify that non-linear DemandMLPRegressor strictly beats Ridge regression MAE on test split."""
    syn = generate_synthetic_demand_series(n_steps=1500, seed=42)
    feats = build_demand_features(syn["power_kw"], voltages=syn["voltage_v"], currents=syn["current_a"])
    X, y, mask = feats["X"], feats["y"], feats["mask"]
    
    split_idx = int(len(X) * 0.7)
    X_train, y_train, mask_train = X[:split_idx], y[:split_idx], mask[:split_idx]
    X_test, y_test, mask_test = X[split_idx:], y[split_idx:], mask[split_idx:]
    
    comp = compare_mlp_and_ridge(
        X_train, y_train,
        X_test, y_test,
        mask_train=mask_train,
        mask_test=mask_test,
        seed=42
    )
    
    assert comp["passed_dod"] is True, f"Demand MLP MAE must beat Ridge: {comp['summary']}"
    assert comp["mlp_metrics"]["mae"] < comp["ridge_metrics"]["mae"]


def test_shallow_model_weight_serialization():
    """Verify saving and loading weights for DemandMLPRegressor and RidgeRegressionBaseline."""
    syn = generate_synthetic_demand_series(n_steps=800, seed=42)
    feats = build_demand_features(syn["power_kw"])
    X, y = feats["X"], feats["y"]
    
    # 1. MLP Regressor serialization
    mlp = DemandMLPRegressor(input_dim=X.shape[1], seed=42)
    mlp.fit(X, y, epochs=10)
    preds_orig = mlp.predict(X[:10])
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_mlp_path = tmp.name
    try:
        mlp.save_weights(tmp_mlp_path)
        
        loaded_mlp = DemandMLPRegressor(input_dim=X.shape[1])
        loaded_mlp.load_weights(tmp_mlp_path)
        preds_loaded = loaded_mlp.predict(X[:10])
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)
    finally:
        if os.path.exists(tmp_mlp_path):
            os.remove(tmp_mlp_path)

    # 2. Ridge Baseline serialization
    ridge = RidgeRegressionBaseline(alpha=1.0)
    ridge.fit(X, y)
    preds_ridge_orig = ridge.predict(X[:10])
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_ridge_path = tmp.name
    try:
        ridge.save(tmp_ridge_path)
        loaded_ridge = RidgeRegressionBaseline()
        loaded_ridge.load(tmp_ridge_path)
        preds_ridge_loaded = loaded_ridge.predict(X[:10])
        np.testing.assert_allclose(preds_ridge_orig, preds_ridge_loaded, rtol=1e-5)
    finally:
        if os.path.exists(tmp_ridge_path):
            os.remove(tmp_ridge_path)
