"""
SmartgridAI Sequence Forecasting Baselines.

Implements:
1. `PersistenceForecaster`: Day-ahead persistence baseline: y_hat_t = y_{t-24h} (t-96).
2. `SeasonalNaiveForecaster`: Week-ahead seasonal naive baseline: y_hat_t = y_{t-168h} (t-672).
3. `LightGBMForecaster`: Multi-output gradient boosted tree tabular regressor on
   calendar features and auto-regressive lags.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class PersistenceForecaster:
    """
    Day-ahead persistence baseline:
    Predicts the consumption from exactly 24 hours prior (96 bins back).
    For a 672-step history window [0:672], the most recent 24h is [576:672].
    """

    def __init__(self, horizon: int = 96) -> None:
        self.horizon = horizon

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Generates day-ahead persistence forecast from history.
        Args:
            history: Array of shape (672,) or (B, 672) or (B, 672, F)
        Returns:
            Array of shape (96,) or (B, 96)
        """
        arr = np.asarray(history, dtype=np.float32)

        # Handle 1D input (672,)
        if arr.ndim == 1:
            if len(arr) < self.horizon:
                raise ValueError(f"History length {len(arr)} is shorter than horizon {self.horizon}.")
            return arr[-self.horizon:].copy()

        # Handle 3D input (B, 672, F) -> extract channel 0 (power)
        if arr.ndim == 3:
            power_slice = arr[:, -self.horizon:, 0]
            return power_slice.copy()

        # Handle 2D input (B, 672)
        if arr.ndim == 2:
            return arr[:, -self.horizon:].copy()

        raise ValueError(f"Unsupported history array shape: {arr.shape}")


class SeasonalNaiveForecaster:
    """
    Week-ahead seasonal-naive baseline:
    Predicts the consumption from exactly 7 days prior (672 bins back).
    For a 672-step history window [0:672], the 24 hours from 7 days ago is [0:96].
    """

    def __init__(self, horizon: int = 96) -> None:
        self.horizon = horizon

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Generates week-ahead seasonal-naive forecast from history.
        Args:
            history: Array of shape (672,) or (B, 672) or (B, 672, F)
        Returns:
            Array of shape (96,) or (B, 96)
        """
        arr = np.asarray(history, dtype=np.float32)

        # Handle 1D input (672,)
        if arr.ndim == 1:
            if len(arr) < self.horizon:
                raise ValueError(f"History length {len(arr)} is shorter than horizon {self.horizon}.")
            return arr[:self.horizon].copy()

        # Handle 3D input (B, 672, F)
        if arr.ndim == 3:
            return arr[:, :self.horizon, 0].copy()

        # Handle 2D input (B, 672)
        if arr.ndim == 2:
            return arr[:, :self.horizon].copy()

        raise ValueError(f"Unsupported history array shape: {arr.shape}")


class LightGBMForecaster:
    """
    Multi-output gradient boosted tabular regressor.
    Extracts auto-regressive lags (t-1, t-2, t-3, t-4, t-96, t-192, t-672),
    summary rolling statistics, and calendar variables to predict the 96 horizon steps.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[Any] = None

    def _extract_tabular_features(self, history_matrix: np.ndarray) -> np.ndarray:
        """
        Extracts tabular features from 672-step history:
        - Recent lags: t-1, t-2, t-3, t-4
        - Daily lags: t-96, t-97, t-192
        - Weekly lag: t-672
        - Rolling stats: mean, std, min, max over last 24h, 48h, 7d
        """
        # If 3D, extract channel 0
        if history_matrix.ndim == 3:
            p_mat = history_matrix[:, :, 0]
        else:
            p_mat = history_matrix

        B, T = p_mat.shape
        feats = []

        # Recent lags (t-1 .. t-4)
        lags = [p_mat[:, -1], p_mat[:, -2], p_mat[:, -3], p_mat[:, -4]]
        # Seasonal lags (t-96, t-192, t-672)
        lags.append(p_mat[:, -96] if T >= 96 else np.zeros(B))
        lags.append(p_mat[:, -192] if T >= 192 else np.zeros(B))
        lags.append(p_mat[:, 0] if T >= 672 else np.zeros(B))

        # Rolling statistics
        r24 = p_mat[:, -96:] if T >= 96 else p_mat
        r168 = p_mat

        feats.extend(lags)
        feats.append(np.nanmean(r24, axis=1))
        feats.append(np.nanstd(r24, axis=1))
        feats.append(np.nanmax(r24, axis=1))
        feats.append(np.nanmin(r24, axis=1))
        feats.append(np.nanmean(r168, axis=1))
        feats.append(np.nanstd(r168, axis=1))

        feat_mat = np.column_stack([np.nan_to_num(f, nan=0.0) for f in feats])
        return feat_mat.astype(np.float32)

    def fit(self, X_hist: np.ndarray, Y_target: np.ndarray) -> "LightGBMForecaster":
        """Fits multi-output LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm and scikit-learn are required for LightGBMForecaster.")

        X_feats = self._extract_tabular_features(X_hist)
        Y_clean = np.nan_to_num(Y_target, nan=0.0)

        base_estimator = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1
        )
        self.model = MultiOutputRegressor(base_estimator)
        self.model.fit(X_feats, Y_clean)
        return self

    def predict(self, X_hist: np.ndarray) -> np.ndarray:
        """Generates predictions for 96 horizon steps."""
        if self.model is None:
            # Fallback to persistence if unfitted
            return PersistenceForecaster().predict(X_hist)

        # Handle 1D
        is_1d = False
        if X_hist.ndim == 1:
            X_hist = X_hist.reshape(1, -1)
            is_1d = True

        X_feats = self._extract_tabular_features(X_hist)
        preds = self.model.predict(X_feats)
        return preds[0] if is_1d else preds


class MajorityClassBaseline:
    """
    Vision majority class baseline predictor.
    Always predicts the most prevalent class seen in training (defaults to 'Straight').
    """

    def __init__(self, default_class: str = "Straight") -> None:
        self.majority_class: str = default_class
        self.classes_: List[str] = ["Straight", "Leaned", "Severe Tilt"]
        self.class_counts_: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, y_train: Sequence[str]) -> "MajorityClassBaseline":
        """Fits majority class baseline on training labels."""
        counts: Dict[str, int] = {}
        for label in y_train:
            counts[label] = counts.get(label, 0) + 1

        self.class_counts_ = counts
        if counts:
            self.majority_class = max(counts.items(), key=lambda kv: kv[1])[0]
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Union[str, List[str]]:
        """Predicts the majority class for all inputs in X."""
        if isinstance(X, (list, tuple, np.ndarray)):
            return [self.majority_class] * len(X)
        return self.majority_class

    def predict_proba(self, X: Any) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Returns probability distribution where the majority class receives probability 1.0.
        """
        single_dist = {c: (1.0 if c == self.majority_class else 0.0) for c in self.classes_}
        if isinstance(X, (list, tuple, np.ndarray)):
            return [single_dist.copy() for _ in range(len(X))]
        return single_dist

    def evaluate(self, y_true: Sequence[str]) -> Dict[str, float]:
        """Evaluates macro-F1 of majority class prediction against y_true."""
        from smartgrid.models.evaluation import compute_macro_f1

        y_pred = [self.majority_class] * len(y_true)
        macro_f1 = compute_macro_f1(y_true, y_pred, classes=self.classes_)
        return {
            "macro_f1": round(macro_f1, 4),
            "baseline_majority_class": self.majority_class
        }

