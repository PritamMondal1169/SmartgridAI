"""
Unit Tests for SmartgridAI Training Dynamics Library (Module M2).
Tests:
- DenseLayer forward, backward, activations, and initializations (Feature 20)
- BatchNorm1D and Dropout1D regularizers (Feature 23)
- FixedDNN 5-layer architecture, forward, backward, and inference (Feature 20)
- Optimizers: SGDMomentum, Adam, RMSprop (Feature 21)
- Layer Gradient-Norm Logger and Global Norm Clipping (Feature 24, 26)
- Sweeps: run_optimizer_sweep, run_initializer_sweep, run_regularizer_sweep (Features 21, 22, 23)
- Vanishing Gradient Reproducer: decay ratio > 100x vs ReLU control (Feature 25)
- Exploding Gradient & Clipping Demo: NaN unclipped divergence vs clipped stability (Feature 26)
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smartgrid.models.dynamics import (
    DenseLayer,
    BatchNorm1D,
    Dropout1D,
    FixedDNN,
    BaseOptimizer,
    SGDMomentumOptimizer,
    AdamOptimizer,
    RMSpropOptimizer,
    run_optimizer_sweep,
    run_initializer_sweep,
    run_regularizer_sweep,
    reproduce_vanishing_gradients,
    reproduce_exploding_gradients_and_clipping
)


# =====================================================================
# Building Blocks: DenseLayer Tests
# =====================================================================

def test_dense_layer_initialization_schemes():
    """Verify weight shapes and variance properties for Random, Xavier, and He initializers."""
    in_dim, out_dim = 100, 50

    # 1. Random Normal (sigma = 0.05)
    layer_rnd = DenseLayer(in_dim, out_dim, init_method="Random", seed=42)
    assert layer_rnd.W.shape == (in_dim, out_dim)
    assert layer_rnd.b.shape == (1, out_dim)
    std_rnd = np.std(layer_rnd.W)
    assert 0.04 < std_rnd < 0.06

    # 2. Xavier / Glorot (std = sqrt(2 / (in + out)) = sqrt(2 / 150) ~= 0.115)
    layer_xav = DenseLayer(in_dim, out_dim, init_method="Xavier", seed=42)
    std_xav = np.std(layer_xav.W)
    expected_xav = np.sqrt(2.0 / (in_dim + out_dim))
    assert np.isclose(std_xav, expected_xav, rtol=0.15)

    # 3. He / Kaiming (std = sqrt(2 / in) = sqrt(2 / 100) ~= 0.141)
    layer_he = DenseLayer(in_dim, out_dim, init_method="He", seed=42)
    std_he = np.std(layer_he.W)
    expected_he = np.sqrt(2.0 / in_dim)
    assert np.isclose(std_he, expected_he, rtol=0.15)

    # 4. Unknown init raises ValueError
    with pytest.raises(ValueError, match="Unknown init_method"):
        DenseLayer(in_dim, out_dim, init_method="InvalidInit")


def test_dense_layer_activations_forward():
    """Verify ReLU, Sigmoid, and Linear forward activations."""
    layer = DenseLayer(2, 2, init_method="He", activation="relu", seed=42)
    # Set explicit weights and biases for predictable output
    layer.W = np.array([[1.0, -1.0], [2.0, -2.0]])
    layer.b = np.array([[0.0, 1.0]])

    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    # z = X W + b
    # row 0: [1*1 + 1*2, 1*-1 + 1*-2] + [0, 1] = [3, -3] + [0, 1] = [3, -2] -> ReLU -> [3, 0]
    # row 1: [-1*1 + -1*2, -1*-1 + -1*-2] + [0, 1] = [-3, 3] + [0, 1] = [-3, 4] -> ReLU -> [0, 4]
    a_relu = layer.forward(X, is_training=False)
    np.testing.assert_allclose(a_relu, [[3.0, 0.0], [0.0, 4.0]])

    # Sigmoid
    layer_sig = DenseLayer(2, 2, activation="sigmoid")
    layer_sig.W = np.zeros((2, 2))
    layer_sig.b = np.zeros((1, 2))
    a_sig = layer_sig.forward(X, is_training=False)
    np.testing.assert_allclose(a_sig, np.full((2, 2), 0.5))

    # Linear
    layer_lin = DenseLayer(2, 2, activation="linear")
    layer_lin.W = layer.W
    layer_lin.b = layer.b
    a_lin = layer_lin.forward(X, is_training=False)
    np.testing.assert_allclose(a_lin, [[3.0, -2.0], [-3.0, 4.0]])

    # Unsupported activation
    layer_bad = DenseLayer(2, 2, activation="bad_act")
    with pytest.raises(ValueError, match="Unsupported activation"):
        layer_bad.forward(X)


def test_dense_layer_backward_pass():
    """Verify analytic backward pass computes correct dW, db, and dX shapes."""
    layer = DenseLayer(3, 2, init_method="He", activation="relu", seed=42)
    X = np.random.randn(5, 3)
    out = layer.forward(X, is_training=True)
    grad_out = np.ones_like(out)

    grad_in = layer.backward(grad_out)
    assert grad_in.shape == X.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape
    assert not np.isnan(layer.dW).any()
    assert not np.isnan(layer.db).any()


def test_dense_layer_backward_without_forward_raises_error():
    """Verify backward pass raises RuntimeError if forward was not called."""
    layer = DenseLayer(3, 2)
    with pytest.raises(RuntimeError, match="Cannot execute backward without prior forward pass"):
        layer.backward(np.ones((2, 2)))


# =====================================================================
# Regularizers: BatchNorm1D and Dropout1D Tests
# =====================================================================

def test_batchnorm1d_forward_and_backward():
    """Verify BatchNorm1D normalizes mini-batches and tracks running statistics."""
    bn = BatchNorm1D(num_features=4, eps=1e-5, momentum=0.1)
    rng = np.random.RandomState(42)
    X = rng.randn(32, 4) * 5.0 + 10.0  # mean ~ 10, std ~ 5

    # 1. Forward training
    out = bn.forward(X, is_training=True)
    assert out.shape == X.shape
    # Batch mean should be ~ 0, batch var ~ 1
    np.testing.assert_allclose(np.mean(out, axis=0), np.zeros(4), atol=1e-4)
    np.testing.assert_allclose(np.var(out, axis=0), np.ones(4), atol=1e-3)

    # Running stats must be updated
    assert not np.allclose(bn.running_mean, 0.0)
    assert not np.allclose(bn.running_var, 1.0)

    # 2. Backward
    grad_out = np.ones_like(out)
    dx = bn.backward(grad_out)
    assert dx.shape == X.shape
    assert bn.dgamma.shape == (1, 4)
    assert bn.dbeta.shape == (1, 4)

    # 3. Forward inference (uses running stats)
    out_eval = bn.forward(X, is_training=False)
    assert out_eval.shape == X.shape


def test_dropout1d_forward_and_backward():
    """Verify Dropout1D masks activations during training and is identity at test time."""
    dropout = Dropout1D(p=0.25, seed=42)
    X = np.ones((100, 100))

    # Forward training: approximately 25% zeros, active units scaled by 1 / (1 - 0.25) = 1.333
    out_train = dropout.forward(X, is_training=True)
    zero_ratio = np.mean(out_train == 0.0)
    assert 0.20 < zero_ratio < 0.30
    non_zeros = out_train[out_train > 0.0]
    np.testing.assert_allclose(non_zeros, 1.0 / 0.75, rtol=1e-4)

    # Backward training: propagates gradient through mask
    grad_out = np.ones_like(out_train)
    dx = dropout.backward(grad_out)
    assert dx.shape == X.shape
    np.testing.assert_allclose(dx == 0.0, out_train == 0.0)

    # Forward evaluation: exact identity
    out_eval = dropout.forward(X, is_training=False)
    np.testing.assert_array_equal(out_eval, X)

    # Backward evaluation: passes gradient unchanged
    dx_eval = dropout.backward(grad_out)
    np.testing.assert_array_equal(dx_eval, grad_out)


# =====================================================================
# Feature 20: Fixed DNN Architecture Tests
# =====================================================================

def test_fixed_dnn_architecture_initialization():
    """Verify FixedDNN creates 5 dense layers with expected hidden dimensions."""
    dnn = FixedDNN(input_dim=10, hidden_dims=[128, 64, 32, 16], output_dim=1, init_method="He")
    assert len(dnn.layers) == 5
    assert dnn.layers[0].in_features == 10
    assert dnn.layers[0].out_features == 128
    assert dnn.layers[1].in_features == 128
    assert dnn.layers[1].out_features == 64
    assert dnn.layers[2].in_features == 64
    assert dnn.layers[2].out_features == 32
    assert dnn.layers[3].in_features == 32
    assert dnn.layers[3].out_features == 16
    assert dnn.layers[4].in_features == 16
    assert dnn.layers[4].out_features == 1
    assert dnn.layers[4].activation == "linear"


def test_fixed_dnn_forward_backward_loss():
    """Verify FixedDNN forward and backward passes compute valid MSE loss and gradients."""
    dnn = FixedDNN(input_dim=8, hidden_dims=[64, 32, 16, 8], output_dim=1, seed=42)
    X = np.random.randn(20, 8)
    y = np.random.randn(20, 1)

    y_pred = dnn.forward(X, is_training=True)
    assert y_pred.shape == (20, 1)

    loss = dnn.backward(y, y_pred)
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not math.isnan(loss)

    norms = dnn.get_layer_gradient_norms()
    assert len(norms) == 5
    assert all(n >= 0.0 for n in norms)


def test_fixed_dnn_fit_and_convergence():
    """Verify FixedDNN fit loop reduces training loss over epochs."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    # Target is linear combination + small non-linearity
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + np.sin(X[:, 2])

    dnn = FixedDNN(input_dim=10, init_method="He", seed=42)
    history = dnn.fit(X, y, epochs=20, batch_size=32, log_gradient_norms=True)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 20
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert len(history["gradient_norms"]) == 20

    # Evaluate predictions
    metrics = dnn.evaluate(X, y)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_fixed_dnn_gradient_clipping():
    """Verify FixedDNN clip_gradients bounds global gradient norm to max_norm."""
    dnn = FixedDNN(input_dim=10, init_method="He", seed=42)
    X = np.random.randn(16, 10)
    y = np.random.randn(16, 1)

    preds = dnn.forward(X, is_training=True)
    dnn.backward(y, preds)

    # Artificially inflate gradients
    for layer in dnn.layers:
        layer.dW *= 100.0
        layer.db *= 100.0

    raw_norm = dnn.clip_gradients(max_norm=1.0)
    assert raw_norm > 1.0

    # Check that clipped Frobenius norm across all layers is bounded by 1.0 + eps
    sum_sq = sum(np.sum(l.dW ** 2) + np.sum(l.db ** 2) for l in dnn.layers)
    clipped_norm = np.sqrt(sum_sq)
    assert clipped_norm <= 1.0 + 1e-6


# =====================================================================
# Feature 21: Optimizer Mechanics & Optimizer Sweep
# =====================================================================

def test_optimizer_step_mechanics():
    """Verify SGD+Momentum, Adam, and RMSprop properly update weights."""
    X = np.ones((4, 5))
    y = np.ones((4, 1)) * 2.0

    # 1. SGD Momentum
    dnn_sgd = FixedDNN(input_dim=5, seed=42)
    w0_sgd = dnn_sgd.layers[0].W.copy()
    opt_sgd = SGDMomentumOptimizer(lr=0.01, momentum=0.9)
    preds = dnn_sgd.forward(X, is_training=True)
    dnn_sgd.backward(y, preds)
    opt_sgd.step(dnn_sgd.layers)
    assert not np.array_equal(dnn_sgd.layers[0].W, w0_sgd)

    # 2. Adam
    dnn_adam = FixedDNN(input_dim=5, seed=42)
    w0_adam = dnn_adam.layers[0].W.copy()
    opt_adam = AdamOptimizer(lr=0.001)
    preds = dnn_adam.forward(X, is_training=True)
    dnn_adam.backward(y, preds)
    opt_adam.step(dnn_adam.layers)
    assert not np.array_equal(dnn_adam.layers[0].W, w0_adam)

    # 3. RMSprop
    dnn_rms = FixedDNN(input_dim=5, seed=42)
    w0_rms = dnn_rms.layers[0].W.copy()
    opt_rms = RMSpropOptimizer(lr=0.001)
    preds = dnn_rms.forward(X, is_training=True)
    dnn_rms.backward(y, preds)
    opt_rms.step(dnn_rms.layers)
    assert not np.array_equal(dnn_rms.layers[0].W, w0_rms)


def test_run_optimizer_sweep():
    """Verify run_optimizer_sweep executes across 3 seeds and returns mean +- std for all optimizers."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 10)
    y_train = rng.randn(80)
    X_val = rng.randn(20, 10)
    y_val = rng.randn(20)

    sweep_results = run_optimizer_sweep(
        X_train, y_train,
        X_val, y_val,
        optimizers=["Adam", "SGD", "RMSprop"],
        seeds=[42, 101, 777],
        epochs=5,
        batch_size=32
    )

    for opt_name in ["Adam", "SGD", "RMSprop"]:
        assert opt_name in sweep_results
        res = sweep_results[opt_name]
        assert "mae_mean" in res
        assert "mae_std" in res
        assert "rmse_mean" in res
        assert "rmse_std" in res
        assert "epochs_to_convergence" in res
        assert "wall_clock_s" in res
        assert res["mae_mean"] >= 0.0
        assert res["mae_std"] >= 0.0
        assert res["wall_clock_s"] >= 0.0


# =====================================================================
# Feature 22: Initializer Sweep
# =====================================================================

def test_run_initializer_sweep():
    """Verify run_initializer_sweep evaluates Random Normal, Xavier, and He initializers."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 10)
    y_train = rng.randn(80)
    X_val = rng.randn(20, 10)
    y_val = rng.randn(20)

    init_results = run_initializer_sweep(
        X_train, y_train,
        X_val, y_val,
        initializers=["Random", "Xavier", "He"],
        epochs=5
    )

    for init_name in ["Random", "Xavier", "He"]:
        assert init_name in init_results
        res = init_results[init_name]
        assert "initial_loss" in res
        assert "initial_gradient_norms" in res
        assert len(res["initial_gradient_norms"]) == 5
        assert "final_mae" in res
        assert "final_rmse" in res


# =====================================================================
# Feature 23: Regularizer Sweep
# =====================================================================

def test_run_regularizer_sweep():
    """Verify run_regularizer_sweep evaluates None, Dropout, and BatchNorm."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 10)
    y_train = rng.randn(80)
    X_val = rng.randn(20, 10)
    y_val = rng.randn(20)

    reg_results = run_regularizer_sweep(
        X_train, y_train,
        X_val, y_val,
        regularizers=["None", "Dropout", "BatchNorm"],
        epochs=5
    )

    for reg_name in ["None", "Dropout", "BatchNorm"]:
        assert reg_name in reg_results
        res = reg_results[reg_name]
        assert "train_mae" in res
        assert "val_mae" in res
        assert "generalization_gap" in res


# =====================================================================
# Feature 25: Vanishing Gradient Reproducer
# =====================================================================

def test_reproduce_vanishing_gradients():
    """
    TEST-T1-MOD-02 & Feature 25:
    Verify vanishing gradient decay ratio > 100x (> 2 orders of magnitude) on 5-layer Sigmoid stack
    and healthy gradient flow on ReLU control.
    """
    result = reproduce_vanishing_gradients(input_dim=10, depth=5, seed=42)

    assert result["passed_dod"] is True
    assert result["decay_ratio"] > 100.0, f"Decay ratio {result['decay_ratio']} must exceed 100.0"
    assert len(result["sigmoid_layer_norms"]) == 5
    assert len(result["relu_control_norms"]) == 5
    # ReLU control gradient ratio should remain healthy (< 10x)
    assert result["relu_ratio"] < 10.0


# =====================================================================
# Feature 26: Exploding Gradient & Norm Clipping Demo
# =====================================================================

def test_reproduce_exploding_gradients_and_clipping():
    """
    TEST-T1-MOD-02 & Feature 26:
    Verify unclipped training under extreme learning rate produces NaN/divergence,
    while gradient norm clipping stabilizes training and prevents NaNs.
    """
    result = reproduce_exploding_gradients_and_clipping(input_dim=10, lr=10.0, clip_norm=1.0, steps=4, seed=42)

    assert result["stabilized"] is True
    assert any(math.isnan(x) for x in result["unclipped_loss_trajectory"]), "Unclipped loss must contain NaN"
    assert not any(math.isnan(x) for x in result["clipped_loss_trajectory"]), "Clipped loss must NOT contain NaN"
    assert result["clipped_loss_trajectory"][0] >= result["clipped_loss_trajectory"][-1]
