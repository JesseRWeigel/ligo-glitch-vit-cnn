"""Paired bootstrap test for comparing two classifiers.

Tests H0: model A's rare-class macro-F1 <= model B's rare-class macro-F1.
Uses identical resampled indices for both models (paired design).

Protocol: >= 10,000 resamples, 95% CI, seed=42.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy, bootstrap_resamples>=10000

import numpy as np
from sklearn.metrics import f1_score


def _rare_class_macro_f1(y_true, y_pred, rare_class_indices):
    """Compute macro-F1 over rare classes only using sklearn.

    Uses sklearn.metrics.f1_score with average='macro' and labels=rare_class_indices.
    This matches the Phase 2 computation exactly.
    """
    return float(f1_score(y_true, y_pred, average="macro", labels=rare_class_indices, zero_division=0))


def _macro_f1(y_true, y_pred):
    """Compute overall macro-F1 using sklearn."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def paired_bootstrap_rare_f1(
    y_true, y_pred_a, y_pred_b, rare_class_indices,
    n_resamples=10000, seed=42, confidence=0.95,
):
    """Paired bootstrap test for rare-class macro-F1 difference.

    For each resample, draws the SAME bootstrap indices for both models,
    then computes the difference in rare-class macro-F1.

    Parameters
    ----------
    y_true : np.ndarray (N,)
        Ground truth labels.
    y_pred_a : np.ndarray (N,)
        Predictions from model A (ViT -- expected to be better).
    y_pred_b : np.ndarray (N,)
        Predictions from model B (CNN -- baseline).
    rare_class_indices : list of int
        Integer indices of rare classes.
    n_resamples : int
        Number of bootstrap resamples (>= 10,000).
    seed : int
        Random seed.
    confidence : float
        Confidence level for CI.

    Returns
    -------
    dict with keys:
        point_estimate_difference : float  (A - B)
        ci_lower : float
        ci_upper : float
        p_value : float  (one-sided, H0: A <= B)
        n_resamples : int
        rare_f1_a : float
        rare_f1_b : float
        bootstrap_deltas : np.ndarray (n_resamples,)
    """
    assert len(y_true) == len(y_pred_a) == len(y_pred_b), "Array lengths must match"
    assert n_resamples >= 10000, f"Need >= 10,000 resamples, got {n_resamples}"

    N = len(y_true)
    rng = np.random.RandomState(seed)

    # Point estimates
    rare_f1_a = _rare_class_macro_f1(y_true, y_pred_a, rare_class_indices)
    rare_f1_b = _rare_class_macro_f1(y_true, y_pred_b, rare_class_indices)
    delta_hat = rare_f1_a - rare_f1_b

    # Bootstrap
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, N, size=N)
        yt = y_true[idx]
        pa = y_pred_a[idx]
        pb = y_pred_b[idx]
        f1_a_i = _rare_class_macro_f1(yt, pa, rare_class_indices)
        f1_b_i = _rare_class_macro_f1(yt, pb, rare_class_indices)
        deltas[i] = f1_a_i - f1_b_i

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # One-sided p-value: fraction where A - B <= 0 (H0: A <= B)
    p_value = float(np.mean(deltas <= 0))

    return {
        "point_estimate_difference": delta_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "rare_f1_a": rare_f1_a,
        "rare_f1_b": rare_f1_b,
        "bootstrap_deltas": deltas,
    }


def paired_bootstrap_metric(
    y_true, y_pred_a, y_pred_b, metric_fn,
    n_resamples=10000, seed=42, confidence=0.95,
):
    """Generic paired bootstrap for any metric function.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred) -> float
        Metric function.

    Returns
    -------
    dict with point_estimate_difference, ci_lower, ci_upper, p_value, metric_a, metric_b.
    """
    N = len(y_true)
    rng = np.random.RandomState(seed)

    metric_a = metric_fn(y_true, y_pred_a)
    metric_b = metric_fn(y_true, y_pred_b)
    delta_hat = metric_a - metric_b

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, N, size=N)
        deltas[i] = metric_fn(y_true[idx], y_pred_a[idx]) - metric_fn(y_true[idx], y_pred_b[idx])

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(deltas <= 0))

    return {
        "point_estimate_difference": delta_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_resamples": n_resamples,
        "metric_a": metric_a,
        "metric_b": metric_b,
    }


def test_paired_bootstrap():
    """Unit test: synthetic data where model A is strictly better on rare classes.

    Model A gets all rare-class predictions right.
    Model B gets all rare-class predictions wrong.
    p-value should be < 0.01.
    """
    rng = np.random.RandomState(123)
    N = 1000
    n_classes = 5
    rare_classes = [0, 1]  # Classes 0 and 1 are rare

    # Generate ground truth: mostly common classes, some rare
    y_true = rng.choice(n_classes, size=N, p=[0.05, 0.05, 0.3, 0.3, 0.3])

    # Model A: perfect on rare, 80% on common
    y_pred_a = y_true.copy()
    common_mask = ~np.isin(y_true, rare_classes)
    flip = rng.random(N) < 0.2
    y_pred_a[common_mask & flip] = rng.choice(n_classes, size=int((common_mask & flip).sum()))

    # Model B: terrible on rare (random), 80% on common
    y_pred_b = y_true.copy()
    rare_mask = np.isin(y_true, rare_classes)
    y_pred_b[rare_mask] = rng.choice(n_classes, size=int(rare_mask.sum()))
    y_pred_b[common_mask & flip] = rng.choice(n_classes, size=int((common_mask & flip).sum()))

    result = paired_bootstrap_rare_f1(
        y_true, y_pred_a, y_pred_b,
        rare_class_indices=rare_classes,
        n_resamples=10000, seed=42,
    )

    assert result["rare_f1_a"] > result["rare_f1_b"], \
        f"Model A should have higher rare-class F1: {result['rare_f1_a']} vs {result['rare_f1_b']}"
    assert result["point_estimate_difference"] > 0, \
        f"Difference should be positive: {result['point_estimate_difference']}"
    assert result["p_value"] < 0.01, \
        f"p-value should be < 0.01 for clearly better model: {result['p_value']}"
    assert result["ci_lower"] > 0, \
        f"95% CI lower bound should be > 0: {result['ci_lower']}"

    print(f"UNIT TEST PASSED:")
    print(f"  rare_f1_a = {result['rare_f1_a']:.4f}")
    print(f"  rare_f1_b = {result['rare_f1_b']:.4f}")
    print(f"  difference = {result['point_estimate_difference']:.4f}")
    print(f"  CI = [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  p-value = {result['p_value']:.6f}")
    return True


if __name__ == "__main__":
    test_paired_bootstrap()
