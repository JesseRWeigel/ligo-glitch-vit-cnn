"""Evaluation utilities for Gravity Spy classification models.

Runs inference on a test set, computes per-class and macro metrics.
Includes consistency check between sklearn and torchmetrics macro-F1.
"""
# ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference on a dataloader, return all predictions and labels as numpy arrays."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels, class_names):
    """Compute classification metrics.

    Returns dict with overall_accuracy, macro_f1, per_class, confusion_matrix.
    Overall accuracy is a SANITY CHECK ONLY, not the primary metric.
    """
    overall_accuracy = float(np.mean(preds == labels))
    macro_f1 = float(f1_score(labels, preds, average="macro"))

    report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        if name in report_dict:
            per_class[name] = {
                "f1": report_dict[name]["f1-score"],
                "recall": report_dict[name]["recall"],
                "precision": report_dict[name]["precision"],
                "support": int(report_dict[name]["support"]),
            }

    cm = confusion_matrix(labels, preds)

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report_str": report_str,
    }


def verify_metric_consistency(preds, labels, num_classes):
    """Verify sklearn macro-F1 matches torchmetrics macro-F1.

    Returns (sklearn_f1, torchmetrics_f1, match: bool, diff: float).
    """
    from torchmetrics.classification import MulticlassF1Score

    sklearn_f1 = float(f1_score(labels, preds, average="macro"))

    tm_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
    tm_f1.update(torch.tensor(preds), torch.tensor(labels))
    torchmetrics_f1 = tm_f1.compute().item()

    diff = abs(sklearn_f1 - torchmetrics_f1)
    match = diff < 1e-6

    return sklearn_f1, torchmetrics_f1, match, diff
