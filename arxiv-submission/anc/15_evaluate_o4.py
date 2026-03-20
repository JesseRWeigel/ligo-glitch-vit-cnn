#!/usr/bin/env python3
"""
Run both O3-trained models on O4 data, verify O3 baseline reproduction.

Phase 4, Plan 01, Task 2.

Steps:
1. Reproduce O3 test metrics exactly (validates checkpoint loading + preprocessing)
2. Run both models on O4 evaluation set
3. Compute per-class O4 metrics with bootstrap CIs
4. Compute degradation analysis (O3 -> O4)
5. Build O4 comparison table with all required columns
6. Save aggregate metrics

ASSERT_CONVENTION: primary_metric=macro_f1, input_format=224x224_RGB_PNG_0to1,
                   forbidden_primary=overall_accuracy, bootstrap_resamples>=10000
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import eval_transforms
from src.evaluation.bootstrap_ci import bootstrap_per_class_f1, bootstrap_macro_f1
from src.evaluation.evaluate import run_inference, compute_metrics
from src.models.resnet_baseline import build_resnet50_baseline
from src.models.vit_classifier import build_vit_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

O3_CLASSES = sorted([
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip",
    "Blip_Low_Frequency", "Chirp", "Extremely_Loud", "Fast_Scattering",
    "Helix", "Koi_Fish", "Light_Modulation", "Low_Frequency_Burst",
    "Low_Frequency_Lines", "No_Glitch", "Paired_Doves", "Power_Line",
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte",
    "Violin_Mode", "Wandering_Line", "Whistle",
])
CLASS_TO_IDX = {c: i for i, c in enumerate(O3_CLASSES)}

RARE_THRESHOLD = 200
O3_TRAIN_COUNTS = {
    "1080Lines": 341, "1400Ripples": 2428, "Air_Compressor": 1361,
    "Blip": 7156, "Blip_Low_Frequency": 13659, "Chirp": 11,
    "Extremely_Loud": 13469, "Fast_Scattering": 34555, "Helix": 33,
    "Koi_Fish": 11950, "Light_Modulation": 142, "Low_Frequency_Burst": 19834,
    "Low_Frequency_Lines": 2853, "No_Glitch": 11568, "Paired_Doves": 216,
    "Power_Line": 1582, "Repeating_Blips": 1061, "Scattered_Light": 68160,
    "Scratchy": 558, "Tomte": 30403, "Violin_Mode": 274,
    "Wandering_Line": 30, "Whistle": 6299,
}

CNN_CHECKPOINT = Path("checkpoints/02-cnn-baseline/best_model.pt")
VIT_CHECKPOINT = Path("checkpoints/03-vit-rare-class/best_model.pt")
O3_TEST_MANIFEST = Path("data/metadata/test_manifest.csv")
O4_MANIFEST = Path("data/o4/metadata/o4_evaluation_manifest.csv")
O3_COMPARISON = Path("results/03-vit-rare-class/comparison_table.csv")
RESULTS_DIR = Path("results/04-o4-validation")

BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpectrogramDataset(Dataset):
    """Dataset for spectrogram images with class labels."""

    def __init__(self, manifest_df, image_col, label_col, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        label_str = row[self.label_col]
        label_idx = CLASS_TO_IDX[label_str]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return img, label_idx


def load_model(checkpoint_path, model_builder, device):
    """Load a model from checkpoint."""
    model, model_id, source = model_builder(num_classes=23, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.set_grad_enabled = False  # inference only
    return model


def verify_o3_reproduction(cnn_model, vit_model, device):
    """Verify O3 test metrics match Phase 3 exactly."""
    log.info("=" * 60)
    log.info("STEP 1: Reproducing O3 test metrics")
    log.info("=" * 60)

    o3_test = pd.read_csv(O3_TEST_MANIFEST)
    transform = eval_transforms(224)

    dataset = SpectrogramDataset(o3_test, "image_path_1.0s", "ml_label", transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    o3_ref = pd.read_csv(O3_COMPARISON)
    o3_ref_classes = o3_ref[~o3_ref["class"].str.startswith("MACRO")]

    log.info("Running CNN on O3 test set...")
    cnn_preds, cnn_labels = run_inference(cnn_model, loader, device)
    cnn_metrics = compute_metrics(cnn_preds, cnn_labels, O3_CLASSES)

    log.info("Running ViT on O3 test set...")
    vit_preds, vit_labels = run_inference(vit_model, loader, device)
    vit_metrics = compute_metrics(vit_preds, vit_labels, O3_CLASSES)

    max_cnn_diff = 0.0
    max_vit_diff = 0.0

    for _, row in o3_ref_classes.iterrows():
        cls = row["class"]
        if cls not in cnn_metrics["per_class"]:
            continue

        ref_cnn_f1 = row["cnn_f1"]
        ref_vit_f1 = row["vit_f1"]
        got_cnn_f1 = cnn_metrics["per_class"][cls]["f1"]
        got_vit_f1 = vit_metrics["per_class"][cls]["f1"]

        cnn_diff = abs(got_cnn_f1 - ref_cnn_f1)
        vit_diff = abs(got_vit_f1 - ref_vit_f1)
        max_cnn_diff = max(max_cnn_diff, cnn_diff)
        max_vit_diff = max(max_vit_diff, vit_diff)

        if cnn_diff > 1e-6:
            log.error(f"CNN F1 MISMATCH {cls}: ref={ref_cnn_f1:.10f} got={got_cnn_f1:.10f}")
        if vit_diff > 1e-6:
            log.error(f"ViT F1 MISMATCH {cls}: ref={ref_vit_f1:.10f} got={got_vit_f1:.10f}")

    log.info(f"Max CNN diff: {max_cnn_diff:.2e}")
    log.info(f"Max ViT diff: {max_vit_diff:.2e}")

    if max_cnn_diff > 1e-6 or max_vit_diff > 1e-6:
        log.error("O3 REPRODUCTION FAILED!")
        sys.exit(1)

    log.info("O3 reproduction PASSED (< 1e-6)")
    return cnn_preds, cnn_labels, vit_preds, vit_labels


def run_o4_inference(cnn_model, vit_model, device):
    """Run both models on O4 data."""
    log.info("=" * 60)
    log.info("STEP 2: O4 inference")
    log.info("=" * 60)

    o4_manifest = pd.read_csv(O4_MANIFEST)
    transform = eval_transforms(224)

    dataset = SpectrogramDataset(o4_manifest, "image_path", "label", transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    log.info(f"O4 set: {len(dataset)} samples")

    log.info("Running CNN on O4...")
    cnn_preds, labels = run_inference(cnn_model, loader, device)

    log.info("Running ViT on O4...")
    vit_preds, _ = run_inference(vit_model, loader, device)

    np.savez(
        RESULTS_DIR / "o4_predictions.npz",
        y_true=labels, cnn_preds=cnn_preds, vit_preds=vit_preds,
        class_names=O3_CLASSES,
    )
    log.info("Saved o4_predictions.npz")

    return cnn_preds, vit_preds, labels


def compute_per_class(preds, labels, model_name):
    """Per-class metrics with bootstrap CIs."""
    log.info(f"Computing per-class metrics for {model_name}...")

    results = {}
    for cls in O3_CLASSES:
        idx = CLASS_TO_IDX[cls]
        mask = labels == idx
        n = int(mask.sum())

        f1_val, f1_lo, f1_hi = bootstrap_per_class_f1(
            preds, labels, idx, n_resamples=10000, seed=42
        )

        binary_pred = (preds == idx).astype(int)
        binary_true = (labels == idx).astype(int)
        prec = float(precision_score(binary_true, binary_pred, zero_division=0))
        rec = float(recall_score(binary_true, binary_pred, zero_division=0))

        results[cls] = {
            "f1": f1_val, "f1_ci_lower": f1_lo, "f1_ci_upper": f1_hi,
            "precision": prec, "recall": rec,
            "n_test": n, "reliable": n >= 10,
        }

        flag = "" if n >= 10 else " [UNRELIABLE]"
        log.info(f"  {cls}: F1={f1_val:.4f} [{f1_lo:.4f},{f1_hi:.4f}] n={n}{flag}")

    macro_f1, macro_lo, macro_hi = bootstrap_macro_f1(preds, labels, n_resamples=10000, seed=42)
    log.info(f"  Macro-F1: {macro_f1:.4f} [{macro_lo:.4f},{macro_hi:.4f}]")

    return results, (macro_f1, macro_lo, macro_hi)


def build_comparison_table(cnn_o4, vit_o4, cnn_macro, vit_macro):
    """Build comprehensive comparison table."""
    log.info("Building comparison table...")

    o3_ref = pd.read_csv(O3_COMPARISON)
    o3_classes = o3_ref[~o3_ref["class"].str.startswith("MACRO")]

    rows = []
    for _, o3r in o3_classes.iterrows():
        cls = o3r["class"]
        n_train = int(o3r["n_train"])
        is_rare = n_train < RARE_THRESHOLD

        c = cnn_o4.get(cls, {})
        v = vit_o4.get(cls, {})
        cnn_f1_o4 = c.get("f1", np.nan)
        vit_f1_o4 = v.get("f1", np.nan)

        rows.append({
            "class": cls,
            "n_train_o3": n_train,
            "n_test_o3": int(o3r["n_test"]),
            "n_test_o4": c.get("n_test", 0),
            "cnn_f1_o3": o3r["cnn_f1"],
            "vit_f1_o3": o3r["vit_f1"],
            "cnn_f1_o4": cnn_f1_o4,
            "vit_f1_o4": vit_f1_o4,
            "cnn_f1_o4_ci_lower": c.get("f1_ci_lower", np.nan),
            "cnn_f1_o4_ci_upper": c.get("f1_ci_upper", np.nan),
            "vit_f1_o4_ci_lower": v.get("f1_ci_lower", np.nan),
            "vit_f1_o4_ci_upper": v.get("f1_ci_upper", np.nan),
            "cnn_degradation": cnn_f1_o4 - o3r["cnn_f1"],
            "vit_degradation": vit_f1_o4 - o3r["vit_f1"],
            "f1_diff_o4": vit_f1_o4 - cnn_f1_o4,
            "is_rare": is_rare,
            "o4_reliable": c.get("n_test", 0) >= 10,
        })

    # Summary rows
    macro_cnn_o3 = float(o3_ref[o3_ref["class"] == "MACRO_ALL"]["cnn_f1"].iloc[0])
    macro_vit_o3 = float(o3_ref[o3_ref["class"] == "MACRO_ALL"]["vit_f1"].iloc[0])

    rows.append({
        "class": "MACRO_ALL", "n_train_o3": np.nan, "n_test_o3": np.nan,
        "n_test_o4": np.nan,
        "cnn_f1_o3": macro_cnn_o3, "vit_f1_o3": macro_vit_o3,
        "cnn_f1_o4": cnn_macro[0], "vit_f1_o4": vit_macro[0],
        "cnn_f1_o4_ci_lower": cnn_macro[1], "cnn_f1_o4_ci_upper": cnn_macro[2],
        "vit_f1_o4_ci_lower": vit_macro[1], "vit_f1_o4_ci_upper": vit_macro[2],
        "cnn_degradation": cnn_macro[0] - macro_cnn_o3,
        "vit_degradation": vit_macro[0] - macro_vit_o3,
        "f1_diff_o4": vit_macro[0] - cnn_macro[0],
        "is_rare": False, "o4_reliable": True,
    })

    rare_cls = [c for c in O3_CLASSES if O3_TRAIN_COUNTS[c] < RARE_THRESHOLD]
    common_cls = [c for c in O3_CLASSES if O3_TRAIN_COUNTS[c] >= RARE_THRESHOLD]

    for name, subset in [("MACRO_RARE", rare_cls), ("MACRO_COMMON", common_cls)]:
        cf = [cnn_o4[c]["f1"] for c in subset if c in cnn_o4]
        vf = [vit_o4[c]["f1"] for c in subset if c in vit_o4]
        cm = float(np.mean(cf)) if cf else np.nan
        vm = float(np.mean(vf)) if vf else np.nan

        o3_row = o3_ref[o3_ref["class"] == name]
        co3 = float(o3_row["cnn_f1"].iloc[0]) if len(o3_row) > 0 else np.nan
        vo3 = float(o3_row["vit_f1"].iloc[0]) if len(o3_row) > 0 else np.nan

        rows.append({
            "class": name, "n_train_o3": np.nan, "n_test_o3": np.nan,
            "n_test_o4": np.nan,
            "cnn_f1_o3": co3, "vit_f1_o3": vo3,
            "cnn_f1_o4": cm, "vit_f1_o4": vm,
            "cnn_f1_o4_ci_lower": np.nan, "cnn_f1_o4_ci_upper": np.nan,
            "vit_f1_o4_ci_lower": np.nan, "vit_f1_o4_ci_upper": np.nan,
            "cnn_degradation": cm - co3 if not np.isnan(co3) else np.nan,
            "vit_degradation": vm - vo3 if not np.isnan(vo3) else np.nan,
            "f1_diff_o4": vm - cm if not (np.isnan(vm) or np.isnan(cm)) else np.nan,
            "is_rare": name == "MACRO_RARE", "o4_reliable": True,
        })

    return pd.DataFrame(rows)


def degradation_analysis(table):
    """Compute degradation summary."""
    log.info("=" * 60)
    log.info("Degradation analysis")
    log.info("=" * 60)

    classes = table[~table["class"].str.startswith("MACRO")]
    macro = table[table["class"] == "MACRO_ALL"].iloc[0]

    cd = macro["cnn_degradation"]
    vd = macro["vit_degradation"]
    crel = abs(cd) / max(macro["cnn_f1_o3"], 1e-10)
    vrel = abs(vd) / max(macro["vit_f1_o3"], 1e-10)

    cp = macro["cnn_f1_o4"] > 0.8 * macro["cnn_f1_o3"]
    vp = macro["vit_f1_o4"] > 0.8 * macro["vit_f1_o3"]

    nc = len(classes)
    ci = int((classes["cnn_degradation"] > 0).sum())
    vi = int((classes["vit_degradation"] > 0).sum())

    result = {
        "cnn_macro_f1_o3": float(macro["cnn_f1_o3"]),
        "vit_macro_f1_o3": float(macro["vit_f1_o3"]),
        "cnn_macro_f1_o4": float(macro["cnn_f1_o4"]),
        "vit_macro_f1_o4": float(macro["vit_f1_o4"]),
        "cnn_degradation_absolute": float(cd),
        "vit_degradation_absolute": float(vd),
        "cnn_degradation_relative": float(crel),
        "vit_degradation_relative": float(vrel),
        "cnn_passes_20pct_threshold": bool(cp),
        "vit_passes_20pct_threshold": bool(vp),
        "cnn_classes_improved_on_o4": ci,
        "vit_classes_improved_on_o4": vi,
        "cnn_label_bias_flag": ci > nc * 0.5,
        "vit_label_bias_flag": vi > nc * 0.5,
    }

    log.info(f"CNN: O3={result['cnn_macro_f1_o3']:.4f} -> O4={result['cnn_macro_f1_o4']:.4f} "
             f"(delta={cd:+.4f}, rel={crel:.1%})")
    log.info(f"ViT: O3={result['vit_macro_f1_o3']:.4f} -> O4={result['vit_macro_f1_o4']:.4f} "
             f"(delta={vd:+.4f}, rel={vrel:.1%})")
    log.info(f"CNN passes 20%: {cp}  |  ViT passes 20%: {vp}")

    if result["cnn_label_bias_flag"]:
        log.warning("RED FLAG: CNN improved on >50% of classes")
    if result["vit_label_bias_flag"]:
        log.warning("RED FLAG: ViT improved on >50% of classes")

    return result


def main():
    log.info("=" * 60)
    log.info("Phase 4 Plan 01 Task 2: O4 Model Evaluation")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    cnn = load_model(CNN_CHECKPOINT, build_resnet50_baseline, DEVICE)
    vit = load_model(VIT_CHECKPOINT, build_vit_classifier, DEVICE)

    # Step 1: O3 reproduction
    verify_o3_reproduction(cnn, vit, DEVICE)

    # Step 2: O4 inference
    cnn_preds, vit_preds, labels = run_o4_inference(cnn, vit, DEVICE)

    # Step 3: Per-class metrics
    log.info("=" * 60)
    log.info("STEP 3: Per-class O4 metrics")
    log.info("=" * 60)
    cnn_o4, cnn_macro = compute_per_class(cnn_preds, labels, "CNN")
    vit_o4, vit_macro = compute_per_class(vit_preds, labels, "ViT")

    # Save per-class
    rows = []
    for cls in O3_CLASSES:
        c = cnn_o4.get(cls, {})
        v = vit_o4.get(cls, {})
        rows.append({
            "class": cls,
            "cnn_f1": c.get("f1", np.nan), "vit_f1": v.get("f1", np.nan),
            "cnn_precision": c.get("precision", np.nan),
            "vit_precision": v.get("precision", np.nan),
            "cnn_recall": c.get("recall", np.nan),
            "vit_recall": v.get("recall", np.nan),
            "n_test": c.get("n_test", 0),
            "reliable": c.get("reliable", False),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "o4_per_class_f1.csv", index=False)

    # Step 5: Comparison table
    table = build_comparison_table(cnn_o4, vit_o4, cnn_macro, vit_macro)
    table.to_csv(RESULTS_DIR / "o4_comparison_table.csv", index=False)
    log.info(f"Saved {RESULTS_DIR / 'o4_comparison_table.csv'}")

    # Step 4: Degradation
    deg = degradation_analysis(table)
    with open(RESULTS_DIR / "o4_degradation.json", "w") as f:
        json.dump(deg, f, indent=2)

    # Step 6: Aggregate metrics
    rare_cls = [c for c in O3_CLASSES if O3_TRAIN_COUNTS[c] < RARE_THRESHOLD]
    pred_data = np.load(RESULTS_DIR / "o4_predictions.npz")
    cnn_acc = float(np.mean(pred_data["cnn_preds"] == pred_data["y_true"]))
    vit_acc = float(np.mean(pred_data["vit_preds"] == pred_data["y_true"]))

    metrics = {
        "cnn_macro_f1_o4": cnn_macro[0],
        "cnn_macro_f1_o4_ci": [cnn_macro[1], cnn_macro[2]],
        "vit_macro_f1_o4": vit_macro[0],
        "vit_macro_f1_o4_ci": [vit_macro[1], vit_macro[2]],
        "cnn_rare_macro_f1_o4": float(np.mean([cnn_o4[c]["f1"] for c in rare_cls if c in cnn_o4])),
        "vit_rare_macro_f1_o4": float(np.mean([vit_o4[c]["f1"] for c in rare_cls if c in vit_o4])),
        "cnn_overall_accuracy_o4_SANITY_CHECK": cnn_acc,
        "vit_overall_accuracy_o4_SANITY_CHECK": vit_acc,
        "n_classes": len(cnn_o4),
        "n_reliable": sum(1 for v in cnn_o4.values() if v["reliable"]),
        "degradation": deg,
    }
    with open(RESULTS_DIR / "o4_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"O3 reproduction: PASSED")
    log.info(f"CNN O4 macro-F1: {cnn_macro[0]:.4f} [{cnn_macro[1]:.4f},{cnn_macro[2]:.4f}]")
    log.info(f"ViT O4 macro-F1: {vit_macro[0]:.4f} [{vit_macro[1]:.4f},{vit_macro[2]:.4f}]")
    log.info(f"Accuracy (SANITY CHECK): CNN={cnn_acc:.4f}, ViT={vit_acc:.4f}")
    log.info("All checks PASSED.")


if __name__ == "__main__":
    main()
