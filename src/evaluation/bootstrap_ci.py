"""Bootstrap confidence intervals for classification metrics.

Uses resampling with replacement per locked protocol Section 3:
- >= 10,000 resamples
- 95% confidence intervals
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, bootstrap_resamples>=10000

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_metric(preds, labels, metric_fn, n_resamples=10000, confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    preds : np.ndarray (N,)
        Predicted class indices.
    labels : np.ndarray (N,)
        True class indices.
    metric_fn : callable(preds, labels) -> float
        Metric function to bootstrap.
    n_resamples : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (0.95 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    point_estimate : float
    ci_lower : float
    ci_upper : float
    bootstrap_distribution : np.ndarray (n_resamples,)
    """
    rng = np.random.RandomState(seed)
    n = len(preds)
    point_estimate = metric_fn(preds, labels)

    bootstrap_values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        bootstrap_values[i] = metric_fn(preds[idx], labels[idx])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper, bootstrap_values


def bootstrap_per_class_f1(preds, labels, class_idx, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap CI for F1 of a single class (binary one-vs-rest).

    Parameters
    ----------
    class_idx : int
        Class index to compute F1 for.

    Returns
    -------
    point_estimate, ci_lower, ci_upper
    """
    def class_f1(p, l):
        # Binary: is this class or not
        binary_pred = (p == class_idx).astype(int)
        binary_true = (l == class_idx).astype(int)
        tp = np.sum(binary_pred & binary_true)
        fp = np.sum(binary_pred & ~binary_true.astype(bool))
        fn = np.sum(~binary_pred.astype(bool) & binary_true)
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    return bootstrap_metric(preds, labels, class_f1, n_resamples, confidence, seed)[:3]


def bootstrap_macro_f1(preds, labels, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap CI for macro-averaged F1."""
    def macro_f1_fn(p, l):
        return float(f1_score(l, p, average="macro", zero_division=0))

    return bootstrap_metric(preds, labels, macro_f1_fn, n_resamples, confidence, seed)[:3]


def bootstrap_per_class_recall(preds, labels, class_idx, n_resamples=10000, confidence=0.95, seed=42):
    """Bootstrap CI for recall of a single class."""
    def class_recall(p, l):
        mask = l == class_idx
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(p[mask] == class_idx))

    return bootstrap_metric(preds, labels, class_recall, n_resamples, confidence, seed)[:3]
