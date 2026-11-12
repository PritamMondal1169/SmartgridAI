"""
SmartgridAI Shallow Neural Network Library (Module M1).
Implements:
- 2D Perceptron Toy Lab (Feature 16) with Rosenblatt learning rule and decision boundary
- XOR 2-2-1 MLP Solver (Feature 17) with BCE loss and multi-seed benchmark
- 15-Minute Demand MLP Regressor (Feature 18) with autoregressive/calendar lags and outage masking
- Closed-Form Ridge Regression Baseline (Feature 19) with L2 regularization
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import math
import json
import pickle
from pathlib import Path


# =====================================================================
# Feature 16: 2D Perceptron Toy Lab
# =====================================================================

def generate_linearly_separable_data(
    n_samples: int = 100,
    seed: int = 42,
    margin: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic 2D linearly separable dataset.
    
    Args:
        n_samples: Total number of 2D data points.
        seed: Pseudorandom generator seed.
        margin: Minimum clearance margin between the two classes.
        
    Returns:
        Tuple of (X, y) where X is shape (n_samples, 2) and y is shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    half = n_samples // 2
    
    # Class 0: clustered below decision line x2 = 1.2 * x1 - margin
    x1_c0 = rng.uniform(-3.0, 3.0, half)
    x2_c0 = 1.2 * x1_c0 - margin - rng.exponential(scale=0.8, size=half)
    y_c0 = np.zeros(half, dtype=int)
    
    # Class 1: clustered above decision line x2 = 1.2 * x1 + margin
    x1_c1 = rng.uniform(-3.0, 3.0, n_samples - half)
    x2_c1 = 1.2 * x1_c1 + margin + rng.exponential(scale=0.8, size=n_samples - half)
    y_c1 = np.ones(n_samples - half, dtype=int)
    
    X = np.vstack([np.column_stack([x1_c0, x2_c0]), np.column_stack([x1_c1, x2_c1])])
    y = np.concatenate([y_c0, y_c1])
    
    # Shuffle samples
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


class Perceptron2D:
    """
    Single-layer Perceptron for 2D binary classification using Frank Rosenblatt's learning rule.
    Activation: Heaviside step function: y_hat = 1 if (w . x + b >= 0) else 0.
    Weight update: w <- w + lr * (y - y_hat) * x, b <- b + lr * (y - y_hat).
    """
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100, seed: Optional[int] = 42):
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.seed = seed
        self.weights: np.ndarray = np.zeros(2, dtype=np.float64)
        self.bias: float = 0.0
        self.converged: bool = False
        self.epochs_trained: int = 0
        self.error_history: List[int] = []
        self.weight_history: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron2D":
        """
        Fits the perceptron to training data X and binary labels y.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        if n_features != 2:
            raise ValueError(f"Perceptron2D requires exactly 2 features, received {n_features}")

        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            self.weights = rng.randn(2) * 0.01
            self.bias = float(rng.randn() * 0.01)
        else:
            self.weights = np.zeros(2, dtype=np.float64)
            self.bias = 0.0

        self.error_history = []
        self.weight_history = []
        self.converged = False

        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                z = np.dot(self.weights, xi) + self.bias
                prediction = 1 if z >= 0.0 else 0
                error = target - prediction
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.error_history.append(errors)
            self.weight_history.append({
                "epoch": epoch,
                "weights": self.weights.tolist(),
                "bias": self.bias,
                "errors": errors
            })

            if errors == 0:
                self.converged = True
                self.epochs_trained = epoch
                return self

        self.epochs_trained = self.max_epochs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class labels {0, 1} for inputs X."""
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0.0, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns classification accuracy on test samples."""
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def get_decision_boundary(self) -> Dict[str, Any]:
        """
        Computes the decision boundary line: w1 * x1 + w2 * x2 + b = 0 => x2 = -(w1/w2) * x1 - (b/w2).
        """
        w1, w2 = self.weights[0], self.weights[1]
        b = self.bias
        if abs(w2) < 1e-12:
            return {
                "slope": float("nan"),
                "intercept": float("nan"),
                "vertical_line_x": -b / (w1 + 1e-12),
                "equation": f"x1 = {-b / (w1 + 1e-12):.4f}"
            }
        slope = -w1 / w2
        intercept = -b / w2
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "equation": f"x2 = {slope:.4f} * x1 + {intercept:.4f}"
        }


# =====================================================================
# Feature 17: XOR 2-2-1 MLP Solver
# =====================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class XORMLPSolver:
    """
    2-2-1 Multi-Layer Perceptron (or 2-1 single layer perceptron) solver for XOR.
    Features:
    - 2 input neurons
    - 2 hidden neurons with sigmoid activation (if hidden=True)
    - 1 output neuron with sigmoid activation
    - Objective: Binary Cross-Entropy (BCE) Loss
    - Optimization: Adam optimizer with multi-start restart mechanism
    - Multi-seed benchmark evaluating convergence DoD across 20 pseudorandom seeds
    - Continuous decision boundary surface evaluation over [-0.5, 1.5]^2
    """
    
    def __init__(
        self,
        hidden: bool = True,
        learning_rate: float = 0.1,
        max_epochs: int = 2500,
        seed: int = 42,
        max_restarts: int = 5
    ):
        self.hidden = bool(hidden)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.seed = int(seed)
        self.max_restarts = int(max_restarts)
        
        # Model parameter state
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        
        self.converged: bool = False
        self.final_loss: float = 1.0
        self.epochs_trained: int = 0
        self.loss_history: List[float] = []

    def _init_weights(self, rng: np.random.RandomState):
        if self.hidden:
            self.W1 = rng.randn(2, 2) * 1.5
            self.b1 = rng.randn(1, 2) * 0.5
            self.W2 = rng.randn(2, 1) * 1.5
            self.b2 = rng.randn(1, 1) * 0.5
        else:
            self.W1 = rng.randn(2, 1) * 1.5
            self.b1 = rng.randn(1, 1) * 0.5
            self.W2 = None
            self.b2 = None

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "XORMLPSolver":
        """
        Trains the XOR network until convergence (BCE loss < 0.01 and 100% accuracy)
        or max_epochs reached across allowed restart attempts.
        """
        if X is None or y is None:
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
            y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        rng = np.random.RandomState(self.seed if self.seed >= 0 else 0)
        self.converged = False
        self.loss_history = []
        total_epochs = 0

        # Number of restarts: with hidden layer we restart if trapped in saddle points;
        # without hidden layer, XOR cannot be solved regardless of restarts.
        restarts = self.max_restarts if self.hidden else 1

        for attempt in range(restarts):
            self._init_weights(rng)
            
            # Adam state buffers
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if self.hidden:
                mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
                mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
                mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
                mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
            else:
                mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
                mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)

            epochs_per_attempt = self.max_epochs // restarts if self.hidden else self.max_epochs

            for ep in range(1, epochs_per_attempt + 1):
                total_epochs += 1
                
                # --- Forward Propagation ---
                if self.hidden:
                    z1 = np.dot(X, self.W1) + self.b1
                    a1 = _sigmoid(z1)
                    z2 = np.dot(a1, self.W2) + self.b2
                    y_hat = _sigmoid(z2)
                else:
                    z1 = np.dot(X, self.W1) + self.b1
                    y_hat = _sigmoid(z1)

                # Binary Cross-Entropy Loss
                eps_bce = 1e-12
                loss = -float(np.mean(y * np.log(y_hat + eps_bce) + (1.0 - y) * np.log(1.0 - y_hat + eps_bce)))
                self.loss_history.append(loss)
                self.final_loss = loss

                # Convergence check: loss < 0.01 and 100% classification accuracy
                preds = (y_hat >= 0.5).astype(int)
                if loss < 0.01 and np.array_equal(preds, y.astype(int)):
                    self.converged = True
                    self.epochs_trained = total_epochs
                    return self

                # --- Backward Propagation ---
                if self.hidden:
                    # Output layer gradient
                    dz2 = (y_hat - y) / float(len(y))
                    dW2 = np.dot(a1.T, dz2)
                    db2 = np.sum(dz2, axis=0, keepdims=True)

                    # Hidden layer gradient
                    da1 = np.dot(dz2, self.W2.T)
                    dz1 = da1 * a1 * (1.0 - a1)
                    dW1 = np.dot(X.T, dz1)
                    db1 = np.sum(dz1, axis=0, keepdims=True)

                    # Adam parameter updates
                    for p, g, m, v in [
                        (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                        (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2)
                    ]:
                        m[:] = beta1 * m + (1.0 - beta1) * g
                        v[:] = beta2 * v + (1.0 - beta2) * (g ** 2)
                        m_h = m / (1.0 - beta1 ** ep)
                        v_h = v / (1.0 - beta2 ** ep)
                        p -= self.learning_rate * m_h / (np.sqrt(v_h) + eps)
                else:
                    dz1 = (y_hat - y) / float(len(y))
                    dW1 = np.dot(X.T, dz1)
                    db1 = np.sum(dz1, axis=0, keepdims=True)

                    for p, g, m, v in [(self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1)]:
                        m[:] = beta1 * m + (1.0 - beta1) * g
                        v[:] = beta2 * v + (1.0 - beta2) * (g ** 2)
                        m_h = m / (1.0 - beta1 ** ep)
                        v_h = v / (1.0 - beta2 ** ep)
                        p -= self.learning_rate * m_h / (np.sqrt(v_h) + eps)

            if self.converged:
                break

        self.epochs_trained = total_epochs
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous output probabilities in [0, 1] for inputs X."""
        X = np.asarray(X, dtype=np.float64)
        if self.W1 is None:
            raise RuntimeError("Model has not been trained yet.")
        if self.hidden:
            z1 = np.dot(X, self.W1) + self.b1
            a1 = _sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            return _sigmoid(z2)
        else:
            z1 = np.dot(X, self.W1) + self.b1
            return _sigmoid(z1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary classification output {0, 1}."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def evaluate_decision_surface(
        self,
        resolution: int = 50,
        x_range: Tuple[float, float] = (-0.5, 1.5),
        y_range: Tuple[float, float] = (-0.5, 1.5)
    ) -> Dict[str, Any]:
        """
        Evaluates the model across a continuous 2D grid [-0.5, 1.5] x [-0.5, 1.5].
        Returns dictionary matching API contract for decision surface visualization.
        """
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        
        proba = self.predict_proba(grid_pts).reshape(resolution, resolution)
        
        return {
            "x": x_vals.tolist(),
            "y": y_vals.tolist(),
            "surface": proba.tolist(),
            "shape": [resolution, resolution]
        }

    @classmethod
    def benchmark_xor_seeds(
        cls,
        n_seeds: int = 20,
        hidden: bool = True
    ) -> Dict[str, Any]:
        """
        Runs XOR solver benchmark across n_seeds pseudorandom seeds.
        DoD: With hidden layer, convergences >= 19/20. Without hidden layer, convergences <= 2/20.
        """
        convergences = 0
        details = []
        for seed in range(n_seeds):
            solver = cls(hidden=hidden, seed=seed)
            solver.fit()
            if solver.converged:
                convergences += 1
            details.append({
                "seed": seed,
                "converged": solver.converged,
                "epochs": solver.epochs_trained,
                "final_loss": round(solver.final_loss, 5)
            })

        return {
            "hidden": hidden,
            "total_seeds": n_seeds,
            "convergences": convergences,
            "convergence_rate": convergences / float(n_seeds),
            "passed_dod": (convergences >= 19) if hidden else (convergences <= 2),
            "details": details
        }


# =====================================================================
# Feature 18: 15-Minute Demand Feature Engineering & MLP Regressor
# =====================================================================

def build_demand_features(
    series: np.ndarray,
    timestamps: Optional[List[Any]] = None,
    voltages: Optional[np.ndarray] = None,
    currents: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Constructs feature matrix for 15-minute demand regression:
    - Autoregressive lags: t-1, t-2, t-3, t-4
    - Daily seasonal lag: t-96 (24 hours * 4 intervals)
    - Weekly seasonal lag: t-672 (7 days * 96 intervals)
    - Rolling statistics: rolling mean (window 4), rolling std (window 4)
    - Cyclical calendar: sin/cos hour, sin/cos day of week
    - Outage mask: True if V <= 5.0 or I <= 0.01

    Returns aligned feature matrix X, target y, and outage mask.
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    max_lag = 672  # Need at least 1 full week of historical context

    if n <= max_lag:
        raise ValueError(f"Series length ({n}) must exceed max lag ({max_lag}) to extract weekly features.")

    # Forward fill NaN values for lag calculation so feature matrix X is finite
    nan_mask = np.isnan(series)
    if np.any(nan_mask):
        idx = np.where(~nan_mask, np.arange(len(series)), 0)
        np.maximum.accumulate(idx, out=idx)
        clean_series = series[idx].copy()
        clean_series[np.isnan(clean_series)] = 0.0
    else:
        clean_series = series

    # Build lags from clean series
    lag1 = clean_series[max_lag - 1 : n - 1]
    lag2 = clean_series[max_lag - 2 : n - 2]
    lag3 = clean_series[max_lag - 3 : n - 3]
    lag4 = clean_series[max_lag - 4 : n - 4]
    lag96 = clean_series[max_lag - 96 : n - 96]
    lag672 = clean_series[0 : n - max_lag]

    # Rolling window stats of prior 4 steps
    rolling_window = np.column_stack([lag1, lag2, lag3, lag4])
    roll_mean = np.mean(rolling_window, axis=1)
    roll_std = np.std(rolling_window, axis=1)

    # Cyclical calendar encodings
    valid_len = n - max_lag
    if timestamps is not None and len(timestamps) == n:
        # Extract datetime components
        hours = []
        dows = []
        for ts in timestamps[max_lag:]:
            if hasattr(ts, "hour"):
                hours.append(ts.hour + ts.minute / 60.0)
                dows.append(ts.weekday())
            else:
                # String timestamp fallback
                hours.append(0.0)
                dows.append(0.0)
        hours = np.array(hours, dtype=np.float64)
        dows = np.array(dows, dtype=np.float64)
    else:
        # Synthesize cyclical indices assuming 15-minute intervals starting at index max_lag
        step_indices = np.arange(max_lag, n)
        hours = ((step_indices * 15) // 60) % 24
        dows = ((step_indices * 15) // (60 * 24)) % 7

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    dow_sin = np.sin(2.0 * np.pi * dows / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dows / 7.0)

    # Feature matrix X
    X = np.column_stack([
        lag1, lag2, lag3, lag4, lag96, lag672,
        roll_mean, roll_std,
        hour_sin, hour_cos, dow_sin, dow_cos
    ])

    # Target y is contemporaneous active power
    y = series[max_lag:]

    # Outage mask: True if voltage <= 5 or current <= 0.01 or power is NaN
    if voltages is not None and currents is not None:
        v_sub = np.asarray(voltages[max_lag:], dtype=np.float64)
        i_sub = np.asarray(currents[max_lag:], dtype=np.float64)
        mask = (v_sub <= 5.0) | (i_sub <= 0.01) | np.isnan(y)
    else:
        mask = np.isnan(y)

    return {
        "X": X,
        "y": y,
        "mask": mask,
        "feature_names": [
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_96", "lag_672",
            "roll_mean_4", "roll_std_4",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
    }


def generate_synthetic_demand_series(
    n_steps: int = 2000,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generates realistic 15-minute synthetic demand time series with:
    - Diurnal 24-hour cycle (peaks at morning and evening)
    - Weekly weekend modulation
    - Non-linear appliance and tariff demand-response dynamics
    - Autoregressive continuity
    - Random grid blackout outages (V <= 5V, I <= 0.01A)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps)
    
    # Diurnal cycle (period = 96 steps)
    hour_val = ((t * 15) // 60) % 24
    h_sin = np.sin(2.0 * np.pi * hour_val / 24.0)
    diurnal = 0.8 + 0.5 * np.sin(2.0 * np.pi * (t % 96) / 96.0 - np.pi / 2.0)
    # Evening secondary peak
    evening = 0.35 * np.exp(-0.5 * (((t % 96) - 76) / 8.0) ** 2)
    # Weekly modulation (period = 672 steps)
    weekly = 1.0 + 0.15 * np.cos(2.0 * np.pi * (t % 672) / 672.0)
    
    # Base load + non-linear harmonics + noise
    base = (diurnal + evening) * weekly
    power = np.zeros(n_steps, dtype=np.float64)
    power[0] = 0.85
    
    for i in range(1, n_steps):
        # Non-linear demand-response threshold:
        # Load switches based on cross-interaction of tariff hours and recent load level
        step_resp = 0.45 if (power[i - 1] > 0.8 and h_sin[i] > 0.0) or (power[i - 1] <= 0.8 and h_sin[i] <= 0.0) else -0.25
        power[i] = 0.45 * power[i - 1] + 0.35 * base[i] + step_resp + rng.randn() * 0.025
        power[i] = max(0.1, min(3.5, power[i]))
    
    voltage = 230.0 + rng.randn(n_steps) * 8.0
    current = (power * 1000.0) / (voltage * 0.9 + 1e-6)
    
    # Inject intermittent outages (3% occurrence)
    outage_prob = rng.rand(n_steps) < 0.03
    voltage[outage_prob] = 0.0
    current[outage_prob] = 0.0
    power[outage_prob] = np.nan
    
    return {
        "power_kw": power,
        "voltage_v": voltage,
        "current_a": current
    }


class DemandMLPRegressor:
    """
    Multi-Layer Perceptron regressor for 15-minute active power forecasting:
    Topology: Input(D) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear).
    Pure NumPy implementation featuring Adam optimization and outage masking.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Tuple[int, int] = (64, 32),
        learning_rate: float = 0.005,
        weight_decay: float = 0.05,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        
        # Initialize weights with He/Kaiming normal initialization
        rng = np.random.RandomState(seed)
        h1, h2 = hidden_dims
        self.W1 = rng.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, h1), dtype=np.float64)
        
        self.W2 = rng.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2), dtype=np.float64)
        
        self.W3 = rng.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, 1), dtype=np.float64)
        
        # Standardization parameters
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
        epochs: int = 60,
        batch_size: int = 64,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
    ) -> Dict[str, List[float]]:
        """
        Trains the demand MLP using Adam with outage masking.
        Rows where mask=True are strictly excluded from the loss calculation and gradients.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            valid_idx = ~mask & ~np.isnan(y.ravel()) & ~np.isnan(X).any(axis=1)
        else:
            valid_idx = ~np.isnan(y.ravel()) & ~np.isnan(X).any(axis=1)

        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Compute standardization parameters from training data only
        self.scaler_mean = np.mean(X_clean, axis=0, keepdims=True)
        self.scaler_std = np.std(X_clean, axis=0, keepdims=True)
        self.scaler_std[self.scaler_std < 1e-8] = 1.0

        self.target_mean = float(np.mean(y_clean))
        self.target_std = float(np.std(y_clean))
        if self.target_std < 1e-8:
            self.target_std = 1.0

        X_norm = (X_clean - self.scaler_mean) / self.scaler_std
        y_norm = (y_clean - self.target_mean) / self.target_std

        n_samples = len(X_clean)
        rng = np.random.RandomState(self.seed)

        # Adam momentum buffers
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        mW3, vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        mb3, vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

        train_loss_history = []
        val_loss_history = []
        step = 0

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            batches = 0

            for i in range(0, n_samples, batch_size):
                step += 1
                idx = perm[i : i + batch_size]
                xb = X_norm[idx]
                yb = y_norm[idx]
                m = len(xb)

                # Forward Pass
                z1 = np.dot(xb, self.W1) + self.b1
                a1 = np.maximum(0.0, z1)  # ReLU
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = np.maximum(0.0, z2)  # ReLU
                z3 = np.dot(a2, self.W3) + self.b3
                y_hat = z3  # Linear

                loss = 0.5 * float(np.mean((yb - y_hat) ** 2))
                epoch_loss += loss
                batches += 1

                # Backward Pass
                dz3 = (y_hat - yb) / float(m)
                dW3 = np.dot(a2.T, dz3) + self.weight_decay * self.W3
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = np.dot(dz3, self.W3.T)
                dz2 = da2 * (z2 > 0.0)
                dW2 = np.dot(a1.T, dz2) + self.weight_decay * self.W2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * (z1 > 0.0)
                dW1 = np.dot(xb.T, dz1) + self.weight_decay * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Adam Updates
                for p, g, mw, vw in [
                    (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2),
                    (self.W3, dW3, mW3, vW3), (self.b3, db3, mb3, vb3),
                ]:
                    mw[:] = beta1 * mw + (1.0 - beta1) * g
                    vw[:] = beta2 * vw + (1.0 - beta2) * (g ** 2)
                    m_h = mw / (1.0 - beta1 ** step)
                    v_h = vw / (1.0 - beta2 ** step)
                    p -= self.learning_rate * m_h / (np.sqrt(v_h) + eps)

            avg_train_loss = epoch_loss / max(1, batches)
            train_loss_history.append(avg_train_loss)

            if val_data is not None:
                val_X, val_y = val_data[0], val_data[1]
                val_mask = val_data[2] if len(val_data) > 2 else None
                val_preds = self.predict(val_X)
                if val_mask is not None:
                    v_valid = ~val_mask & ~np.isnan(val_y.ravel())
                    v_loss = float(np.mean((val_y[v_valid] - val_preds[v_valid]) ** 2))
                else:
                    v_loss = float(np.mean((val_y - val_preds) ** 2))
                val_loss_history.append(v_loss)

        return {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates real-valued power demand predictions in kW."""
        X = np.asarray(X, dtype=np.float64)
        if self.scaler_mean is None or self.scaler_std is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Standardize features
        X_norm = (X - self.scaler_mean) / self.scaler_std
        # Forward pass
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0.0, z2)
        z3 = np.dot(a2, self.W3) + self.b3
        
        # Invert target scaling
        y_pred = z3 * self.target_std + self.target_mean
        return np.maximum(0.0, y_pred.ravel())

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Computes MAE, RMSE, and R2 on valid (unmasked) evaluation intervals."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if mask is not None:
            valid = ~mask & ~np.isnan(y)
        else:
            valid = ~np.isnan(y)
            
        y_true = y[valid]
        y_hat = preds[valid]
        
        if len(y_true) == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            
        mae = float(np.mean(np.abs(y_true - y_hat)))
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
        
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def save_weights(self, filepath: Union[str, Path]):
        """Saves weights and normalization parameters to file."""
        payload = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def load_weights(self, filepath: Union[str, Path]):
        """Loads weights and normalization parameters from file."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.W1 = payload["W1"]
        self.b1 = payload["b1"]
        self.W2 = payload["W2"]
        self.b2 = payload["b2"]
        self.W3 = payload["W3"]
        self.b3 = payload["b3"]
        self.scaler_mean = payload["scaler_mean"]
        self.scaler_std = payload["scaler_std"]
        self.target_mean = payload["target_mean"]
        self.target_std = payload["target_std"]


# =====================================================================
# Feature 19: Ridge Regression Baseline
# =====================================================================

class RidgeRegressionBaseline:
    """
    Closed-Form L2-Regularized Linear Regression Baseline:
    Solves w_ridge = (X_aug^T X_aug + alpha * I)^(-1) X_aug^T y.
    Serves as the direct linear benchmark against which DemandMLPRegressor is verified.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.weights: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> "RidgeRegressionBaseline":
        """Computes closed-form L2 regularized regression coefficients."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            valid = ~mask & ~np.isnan(y.ravel()) & ~np.isnan(X).any(axis=1)
        else:
            valid = ~np.isnan(y.ravel()) & ~np.isnan(X).any(axis=1)

        X_clean = X[valid]
        y_clean = y[valid]

        self.scaler_mean = np.mean(X_clean, axis=0, keepdims=True)
        self.scaler_std = np.std(X_clean, axis=0, keepdims=True)
        self.scaler_std[self.scaler_std < 1e-8] = 1.0

        X_norm = (X_clean - self.scaler_mean) / self.scaler_std
        
        # Augment with bias column of ones
        n_samples, n_features = X_norm.shape
        X_aug = np.column_stack([np.ones((n_samples, 1)), X_norm])
        
        # Regularization matrix: do not penalize bias (index 0)
        reg_matrix = self.alpha * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0.0

        # Normal equation: (X^T X + alpha * I)^(-1) X^T y
        A = np.dot(X_aug.T, X_aug) + reg_matrix
        b = np.dot(X_aug.T, y_clean)
        self.weights = np.linalg.solve(A, b).ravel()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts active power in kW using fitted Ridge coefficients."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.scaler_mean) / self.scaler_std
        n_samples = len(X_norm)
        X_aug = np.column_stack([np.ones((n_samples, 1)), X_norm])
        preds = np.dot(X_aug, self.weights)
        return np.maximum(0.0, preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluates baseline MAE, RMSE, and R2."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if mask is not None:
            valid = ~mask & ~np.isnan(y)
        else:
            valid = ~np.isnan(y)
        y_true = y[valid]
        y_hat = preds[valid]
        
        mae = float(np.mean(np.abs(y_true - y_hat)))
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_hat) ** 2))
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def save(self, filepath: Union[str, Path]):
        """Serializes Ridge baseline to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "weights": self.weights,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std
            }, f)

    def load(self, filepath: Union[str, Path]):
        """Loads serialized Ridge baseline from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.alpha = data["alpha"]
        self.weights = data["weights"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]


# =====================================================================
# Comparative Verification Benchmark
# =====================================================================

def compare_mlp_and_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mask_train: Optional[np.ndarray] = None,
    mask_test: Optional[np.ndarray] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains both DemandMLPRegressor and RidgeRegressionBaseline on identical
    feature matrices and verifies that MLP MAE < Ridge MAE on the test split.
    """
    mlp = DemandMLPRegressor(input_dim=X_train.shape[1], learning_rate=0.005, weight_decay=0.05, seed=seed)
    mlp.fit(X_train, y_train, mask=mask_train, epochs=80)
    mlp_metrics = mlp.evaluate(X_test, y_test, mask=mask_test)

    ridge = RidgeRegressionBaseline(alpha=1.0)
    ridge.fit(X_train, y_train, mask=mask_train)
    ridge_metrics = ridge.evaluate(X_test, y_test, mask=mask_test)

    passed_dod = mlp_metrics["mae"] < ridge_metrics["mae"]
    return {
        "mlp_metrics": mlp_metrics,
        "ridge_metrics": ridge_metrics,
        "mae_delta": round(ridge_metrics["mae"] - mlp_metrics["mae"], 5),
        "passed_dod": passed_dod,
        "summary": (
            f"MLP MAE ({mlp_metrics['mae']:.4f}) {'<' if passed_dod else '>='} "
            f"Ridge MAE ({ridge_metrics['mae']:.4f}) -> {'PASSED' if passed_dod else 'FAILED'}"
        )
    }
