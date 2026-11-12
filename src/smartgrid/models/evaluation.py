"""
SmartgridAI Model Evaluation & Metric Engine.

Implements:
1. `compute_mase`: Scale-free Mean Absolute Scaled Error using in-sample seasonal
   training benchmark with division-by-zero protection.
2. `compute_mae`, `compute_rmse`, `compute_mape`: Standard time series error metrics
   with outage mask support.
3. `compute_metrics_dict`: Unified evaluation metric suite.
4. `evaluate_portfolio_beat_rate`: Calculates the fraction of admitted meters
   beating the Seasonal-Naive benchmark (DoD: >= 70% of admitted meters).
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import math
import numpy as np


def compute_mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_actual: np.ndarray,
    seasonality: int = 96
) -> float:
    """
    Computes scale-free Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Args:
        actual: Held-out test ground truth array (H,).
        predicted: Model forecast array (H,).
        train_actual: In-sample training series array (T,).
        seasonality: Seasonal period S (default 96 for 24h @ 15 min).

    Returns:
        Scale-free MASE float. Values < 1.0 indicate superior performance
        relative to the in-sample Seasonal-Naive benchmark.
    """
    act = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    train = np.asarray(train_actual, dtype=np.float64)

    # Filter out NaNs
    valid_test = ~np.isnan(act) & ~np.isnan(pred)
    if not np.any(valid_test):
        return 0.0

    mae_model = np.mean(np.abs(act[valid_test] - pred[valid_test]))

    # Compute in-sample seasonal naive scale
    clean_train = train[~np.isnan(train)]
    if len(clean_train) <= seasonality:
        diffs = np.abs(np.diff(clean_train))
    else:
        diffs = np.abs(clean_train[seasonality:] - clean_train[:-seasonality])

    if len(diffs) == 0:
        scale = 0.0
    else:
        scale = float(np.mean(diffs))

    # Flatline / zero-scale division-by-zero protection
    if scale == 0.0 or np.isnan(scale):
        return float("inf") if mae_model > 0.0 else 1.0

    return float(mae_model / scale)


def compute_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Computes Mean Absolute Error, optionally masked at outage intervals."""
    act = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    valid = ~np.isnan(act) & ~np.isnan(pred)
    if mask is not None:
        valid = valid & (np.asarray(mask) > 0.5)

    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs(act[valid] - pred[valid])))


def compute_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Computes Root Mean Squared Error, optionally masked at outage intervals."""
    act = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    valid = ~np.isnan(act) & ~np.isnan(pred)
    if mask is not None:
        valid = valid & (np.asarray(mask) > 0.5)

    if not np.any(valid):
        return 0.0
    return float(np.sqrt(np.mean((act[valid] - pred[valid]) ** 2)))


def compute_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-5
) -> float:
    """
    Computes Mean Absolute Percentage Error with epsilon smoothing
    to prevent division-by-zero during sub-watt standby power.
    """
    act = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)

    valid = ~np.isnan(act) & ~np.isnan(pred)
    if mask is not None:
        valid = valid & (np.asarray(mask) > 0.5)

    if not np.any(valid):
        return 0.0

    denom = np.abs(act[valid]) + eps
    mape = np.mean(np.abs(act[valid] - pred[valid]) / denom) * 100.0
    return float(mape)


def compute_metrics_dict(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_actual: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    seasonality: int = 96
) -> Dict[str, float]:
    """Computes aggregate metrics bundle: MAE, RMSE, MAPE, and MASE."""
    metrics = {
        "mae": round(compute_mae(actual, predicted, mask=mask), 4),
        "rmse": round(compute_rmse(actual, predicted, mask=mask), 4),
        "mape": round(compute_mape(actual, predicted, mask=mask), 2)
    }
    if train_actual is not None:
        metrics["mase"] = round(compute_mase(actual, predicted, train_actual, seasonality=seasonality), 4)
    return metrics


def evaluate_portfolio_beat_rate(
    per_meter_mase: Dict[str, float],
    benchmark_threshold: float = 1.0,
    target_beat_rate: float = 0.70
) -> Dict[str, Any]:
    """
    Evaluates whether forecasting models beat Seasonal-Naive MASE across >= 70% of admitted meters.

    Args:
        per_meter_mase: Dictionary mapping meter_id -> MASE score.
        benchmark_threshold: Seasonal-Naive MASE threshold (default 1.0).
        target_beat_rate: DoD requirement threshold (default 0.70).

    Returns:
        Dictionary with beat counts, beat rate, and DoD acceptance verdict.
    """
    if not per_meter_mase:
        return {
            "beat_count": 0,
            "total_meters": 0,
            "beat_rate": 0.0,
            "meets_dod": False
        }

    beat_count = sum(1 for score in per_meter_mase.values() if score < benchmark_threshold)
    total_meters = len(per_meter_mase)
    beat_rate = beat_count / float(total_meters)

    return {
        "beat_count": beat_count,
        "total_meters": total_meters,
        "beat_rate": round(beat_rate, 4),
        "target_beat_rate": target_beat_rate,
        "meets_dod": beat_rate >= target_beat_rate
    }


def compute_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Optional[Sequence[str]] = None
) -> float:
    """
    Computes macro-averaged F1 score across all classes.

    Args:
        y_true: Ground truth class label sequence.
        y_pred: Predicted class label sequence.
        classes: Optional list of class names. If None, inferred from unique labels.

    Returns:
        Macro-averaged F1 score in [0.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")
    if len(y_true) == 0:
        return 0.0

    if classes is None:
        cls_set = sorted(list(set(y_true) | set(y_pred)))
    else:
        cls_set = list(classes)

    if not cls_set:
        return 0.0

    f1_scores: List[float] = []
    for c in cls_set:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def _box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Computes Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_map50(
    gt_boxes: Union[List[List[float]], List[List[List[float]]]],
    pred_boxes: Union[List[List[float]], List[List[List[float]]]],
    pred_scores: Optional[Union[List[float], List[List[float]]]] = None,
    iou_threshold: float = 0.5
) -> float:
    """
    Computes mean Average Precision at IoU >= 0.5 (mAP@0.5).

    Args:
        gt_boxes: Ground truth boxes per image, or single image box list.
        pred_boxes: Predicted boxes per image, or single image box list.
        pred_scores: Optional confidence scores per prediction.
        iou_threshold: Minimum IoU for true positive match (default 0.5).

    Returns:
        mAP@0.5 float in [0.0, 1.0].
    """
    # Normalize inputs to list of images
    if not gt_boxes and not pred_boxes:
        return 1.0

    # Check if single-image format: list of boxes [[x1, y1, x2, y2], ...]
    if gt_boxes and isinstance(gt_boxes[0], (list, tuple, np.ndarray)) and (not gt_boxes[0] or not isinstance(gt_boxes[0][0], (list, tuple, np.ndarray))):
        gt_list: List[List[List[float]]] = [gt_boxes]  # type: ignore
        pred_list: List[List[List[float]]] = [pred_boxes]  # type: ignore
        score_list: Optional[List[List[float]]] = [pred_scores] if pred_scores is not None else None  # type: ignore
    else:
        gt_list = gt_boxes  # type: ignore
        pred_list = pred_boxes  # type: ignore
        score_list = pred_scores  # type: ignore

    total_gts = sum(len(g) for g in gt_list)
    if total_gts == 0:
        return 1.0 if sum(len(p) for p in pred_list) == 0 else 0.0

    # Collect all predictions with image index and confidence score
    flat_preds = []
    for img_idx, p_boxes in enumerate(pred_list):
        scores = score_list[img_idx] if score_list is not None and img_idx < len(score_list) and score_list[img_idx] is not None else [1.0] * len(p_boxes)
        for b, s in zip(p_boxes, scores):
            flat_preds.append((img_idx, b, float(s)))

    # Sort predictions by score descending
    flat_preds.sort(key=lambda x: x[2], reverse=True)

    # Track matched ground truth boxes per image
    matched_gts: Dict[int, set] = {i: set() for i in range(len(gt_list))}
    tp = np.zeros(len(flat_preds), dtype=np.float64)
    fp = np.zeros(len(flat_preds), dtype=np.float64)

    for p_idx, (img_idx, pred_b, _) in enumerate(flat_preds):
        img_gts = gt_list[img_idx] if img_idx < len(gt_list) else []
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_b in enumerate(img_gts):
            iou = _box_iou(pred_b, gt_b)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gts[img_idx]:
            tp[p_idx] = 1.0
            matched_gts[img_idx].add(best_gt_idx)
        else:
            fp[p_idx] = 1.0

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gts
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # 11-point interpolated AP calculation
    ap = 0.0
    for r_thresh in np.linspace(0.0, 1.0, 11):
        prec_at_r = precisions[recalls >= r_thresh]
        if len(prec_at_r) > 0:
            ap += np.max(prec_at_r)
    return float(ap / 11.0)


def compute_miou(
    y_true_mask: np.ndarray,
    y_pred_mask: np.ndarray,
    num_classes: int = 3
) -> float:
    """
    Computes mean Intersection over Union (mIoU) across semantic segmentation classes.

    Args:
        y_true_mask: Ground truth integer mask of shape (H, W) or (B, H, W).
        y_pred_mask: Predicted integer mask of shape (H, W) or (B, H, W).
        num_classes: Number of distinct semantic classes (default 3: 0: bg, 1: conductor, 2: vegetation).

    Returns:
        Mean IoU float in [0.0, 1.0].
    """
    yt = np.asarray(y_true_mask, dtype=np.int64)
    yp = np.asarray(y_pred_mask, dtype=np.int64)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true_mask {yt.shape} vs y_pred_mask {yp.shape}")

    ious: List[float] = []
    for c in range(num_classes):
        inter = np.sum((yt == c) & (yp == c))
        union = np.sum((yt == c) | (yp == c))
        if union == 0:
            # If class is absent from both ground truth and prediction, IoU is 1.0
            ious.append(1.0)
        else:
            ious.append(float(inter / union))

    return float(np.mean(ious))


def compute_latency_percentile(
    latencies_ms: Sequence[float],
    percentile: float = 95.0
) -> float:
    """Computes the requested percentile of an array of inference latencies."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, percentile))

