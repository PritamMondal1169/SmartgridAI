"""
SmartgridAI Machine Learning Models Package.
Exports M1 Shallow, M2 Dynamics, M3 Vision, M4 Sequence, and Evaluation/Baselines.
"""

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

from smartgrid.models.vision import (
    validate_and_sanitize_image,
    VisionError,
    ImageValidationError,
    InvalidImageFormatError,
    PayloadTooLargeError,
    ImageDimensionError,
    CorruptedImageError,
    PoleLeanCNN,
    VegetationConductorUNet,
    PoleLeanClassifier,
    PoleDetector,
    VegetationSegmenter,
    VisionInspectionPipeline
)

from smartgrid.models.baselines import (
    MajorityClassBaseline,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    LightGBMForecaster
)

from smartgrid.models.evaluation import (
    compute_macro_f1,
    compute_map50,
    compute_miou,
    compute_latency_percentile,
    compute_mase,
    compute_mae,
    compute_rmse,
    compute_mape,
    compute_metrics_dict,
    evaluate_portfolio_beat_rate
)

__all__ = [
    # M1 Shallow
    "generate_linearly_separable_data",
    "Perceptron2D",
    "XORMLPSolver",
    "build_demand_features",
    "generate_synthetic_demand_series",
    "DemandMLPRegressor",
    "RidgeRegressionBaseline",
    "compare_mlp_and_ridge",
    # M2 Dynamics
    "DenseLayer",
    "BatchNorm1D",
    "Dropout1D",
    "FixedDNN",
    "BaseOptimizer",
    "SGDMomentumOptimizer",
    "AdamOptimizer",
    "RMSpropOptimizer",
    "run_optimizer_sweep",
    "run_initializer_sweep",
    "run_regularizer_sweep",
    "reproduce_vanishing_gradients",
    "reproduce_exploding_gradients_and_clipping",
    # M3 Vision
    "validate_and_sanitize_image",
    "VisionError",
    "ImageValidationError",
    "InvalidImageFormatError",
    "PayloadTooLargeError",
    "ImageDimensionError",
    "CorruptedImageError",
    "PoleLeanCNN",
    "VegetationConductorUNet",
    "PoleLeanClassifier",
    "PoleDetector",
    "VegetationSegmenter",
    "VisionInspectionPipeline",
    # Baselines
    "MajorityClassBaseline",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "LightGBMForecaster",
    # Evaluation
    "compute_macro_f1",
    "compute_map50",
    "compute_miou",
    "compute_latency_percentile",
    "compute_mase",
    "compute_mae",
    "compute_rmse",
    "compute_mape",
    "compute_metrics_dict",
    "evaluate_portfolio_beat_rate"
]
