#!/usr/bin/env python3
"""
Test threshold persistence on O4 and produce deliverable figures.

Phase 4, Plan 01, Task 3.

Steps:
1. Spearman rank correlation: n_train_o3 vs (vit_f1 - cnn_f1) on O4
2. Sign test for 100+ sample classes
3. Sensitivity analysis across minimum O4 sample thresholds
4. O3 cross-check (reproduce Phase 3 finding)
5. Threshold scatter plot (deliv-threshold-scatter)
6. Degradation bar chart (deliv-degradation-plot)

ASSERT_CONVENTION: primary_metric=macro_f1, forbidden_primary=overall_accuracy,
                   bootstrap_resamples>=10000
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results/04-o4-validation")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SAMPLE_EFFICIENCY_THRESHOLD = 100  # ~100 O3 training samples


def load_comparison_table():
    """Load the O4 comparison table from Task 2."""
    df = pd.read_csv(RESULTS_DIR / "o4_comparison_table.csv")
    # Filter out summary rows
    class_df = df[~df["class"].isin(["MACRO_ALL", "MACRO_RARE", "MACRO_COMMON"])].copy()
    class_df["n_train_o3"] = class_df["n_train_o3"].astype(int)
    class_df["n_test_o4"] = class_df["n_test_o4"].astype(int)
    return class_df


def spearman_test(df, min_o4_samples=10):
    """Compute Spearman rank correlation between n_train_o3 and f1_diff_o4."""
    reliable = df[df["n_test_o4"] >= min_o4_samples].copy()
    n_classes = len(reliable)
    if n_classes < 5:
        return {
            "rho": None, "p_value": None, "n_classes": n_classes,
            "min_o4_samples": min_o4_samples, "status": "insufficient_data"
        }
    rho, p = stats.spearmanr(reliable["n_train_o3"], reliable["f1_diff_o4"])
    return {
        "rho": float(rho), "p_value": float(p), "n_classes": int(n_classes),
        "min_o4_samples": int(min_o4_samples),
        "status": "significant" if (rho > 0 and p < 0.05) else
                  "suggestive" if (rho > 0 and p < 0.1) else "not_confirmed"
    }


def sign_test(df, min_train=100, min_o4_samples=10):
    """Sign test: among 100+ train classes with reliable O4 data, does ViT win more?"""
    subset = df[(df["n_train_o3"] >= min_train) & (df["n_test_o4"] >= min_o4_samples)].copy()
    n_total = len(subset)
    if n_total < 3:
        return {
            "n_positive": 0, "n_negative": 0, "n_zero": 0, "n_total": n_total,
            "p_value": None, "status": "insufficient_data"
        }
    n_positive = int((subset["f1_diff_o4"] > 0).sum())
    n_negative = int((subset["f1_diff_o4"] < 0).sum())
    n_zero = int((subset["f1_diff_o4"] == 0).sum())
    # Binomial test: H0 = equal probability of ViT/CNN winning
    p_val = float(stats.binomtest(n_positive, n_positive + n_negative, 0.5).pvalue)
    return {
        "n_positive": n_positive, "n_negative": n_negative, "n_zero": n_zero,
        "n_total": int(n_total), "p_value": p_val,
        "vit_win_rate": n_positive / max(n_positive + n_negative, 1),
        "status": "vit_majority" if n_positive > n_negative else "cnn_majority"
    }


def sensitivity_analysis(df):
    """Vary minimum O4 sample threshold and recompute Spearman rho."""
    thresholds = [5, 10, 15, 20, 30]
    results = []
    for thresh in thresholds:
        res = spearman_test(df, min_o4_samples=thresh)
        results.append(res)
    return results


def o3_cross_check(df):
    """Cross-check: Spearman on O3 data should reproduce Phase 3 positive correlation."""
    # Use O3 F1 differences
    f1_diff_o3 = df["vit_f1_o3"] - df["cnn_f1_o3"]
    rho, p = stats.spearmanr(df["n_train_o3"], f1_diff_o3)
    return {
        "rho": float(rho), "p_value": float(p), "n_classes": int(len(df)),
        "note": "Cross-check using O3 test set F1 differences"
    }


def make_threshold_scatter(df, spearman_result, output_path):
    """Scatter plot: N_train (O3) vs F1 diff on O4."""
    fig, ax = plt.subplots(figsize=(10, 7))

    reliable = df[df["o4_reliable"] == True]
    unreliable = df[df["o4_reliable"] == False]

    # Plot unreliable points (faded)
    if len(unreliable) > 0:
        ax.scatter(unreliable["n_train_o3"], unreliable["f1_diff_o4"],
                   c="gray", alpha=0.3, s=60, zorder=2, label="Unreliable (N_O4 < 10)")

    # Plot reliable points
    ax.scatter(reliable["n_train_o3"], reliable["f1_diff_o4"],
               c="steelblue", s=80, zorder=3, edgecolors="black", linewidths=0.5,
               label=f"Reliable (N_O4 ≥ 10, n={len(reliable)})")

    # Label outlier classes (|f1_diff| > 0.15 or extreme N_train)
    for _, row in df.iterrows():
        if abs(row["f1_diff_o4"]) > 0.15 or row["n_train_o3"] < 30:
            offset = (5, 5) if row["f1_diff_o4"] >= 0 else (5, -12)
            ax.annotate(row["class"], (row["n_train_o3"], row["f1_diff_o4"]),
                        fontsize=7, textcoords="offset points", xytext=offset,
                        alpha=0.8)

    # Threshold line at N_train = 100
    ax.axvline(x=SAMPLE_EFFICIENCY_THRESHOLD, color="red", linestyle="--",
               alpha=0.6, linewidth=1.5, label=f"Threshold (~{SAMPLE_EFFICIENCY_THRESHOLD} samples)")

    # Parity line at f1_diff = 0
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("O3 Training Set Size (N_train)", fontsize=12)
    ax.set_ylabel("ViT F1 − CNN F1 (on O4)", fontsize=12)
    ax.set_title("Sample-Efficiency Threshold: O4 Validation", fontsize=14)

    # Annotate Spearman result
    rho = spearman_result["rho"]
    p = spearman_result["p_value"]
    n = spearman_result["n_classes"]
    if rho is not None:
        ax.text(0.02, 0.98, f"Spearman ρ = {rho:.3f}, p = {p:.4f} (n = {n})",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved threshold scatter: {output_path}")


def make_degradation_chart(df, output_path):
    """Grouped bar chart: per-class F1 degradation (O3->O4) for both models."""
    # Sort by n_train_o3 ascending
    df_sorted = df.sort_values("n_train_o3").copy()

    classes = df_sorted["class"].tolist()
    cnn_deg = df_sorted["cnn_degradation"].values
    vit_deg = df_sorted["vit_degradation"].values
    is_rare = df_sorted["is_rare"].values

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_cnn = ax.bar(x - width/2, cnn_deg, width, label="CNN (ResNet-50)",
                      color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.3)
    bars_vit = ax.bar(x + width/2, vit_deg, width, label="ViT-B/16",
                      color="darkorange", alpha=0.8, edgecolor="black", linewidth=0.3)

    # Highlight rare classes
    for i, rare in enumerate(is_rare):
        if rare:
            for bar_group in [bars_cnn, bars_vit]:
                bar = bar_group[i]
                bar.set_edgecolor("red")
                bar.set_linewidth(2)

    # -20% degradation threshold line
    ax.axhline(y=-0.20, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
               label="−20% contract threshold")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    ax.set_xlabel("Glitch Class (sorted by O3 training set size →)", fontsize=11)
    ax.set_ylabel("F1 Degradation (O4 − O3)", fontsize=11)
    ax.set_title("Per-Class F1 Degradation: O3 → O4", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved degradation chart: {output_path}")


def main():
    log.info("=== Task 3: Threshold persistence test ===")

    df = load_comparison_table()
    log.info(f"Loaded {len(df)} classes from comparison table")

    # Step 1: Spearman test (primary)
    log.info("--- Spearman rank correlation test ---")
    spearman = spearman_test(df, min_o4_samples=10)
    log.info(f"Spearman rho={spearman['rho']:.4f}, p={spearman['p_value']:.4f}, "
             f"n={spearman['n_classes']}, status={spearman['status']}")

    # Step 2: Sign test for 100+ sample classes
    log.info("--- Sign test (N_train >= 100) ---")
    sign = sign_test(df, min_train=100, min_o4_samples=10)
    log.info(f"Sign test: {sign['n_positive']}/{sign['n_total']} classes ViT > CNN, "
             f"p={sign['p_value']:.4f}, status={sign['status']}")

    # Step 3: Sensitivity analysis
    log.info("--- Sensitivity analysis ---")
    sensitivity = sensitivity_analysis(df)
    for s in sensitivity:
        if s["rho"] is not None:
            log.info(f"  min_O4={s['min_o4_samples']}: rho={s['rho']:.4f}, "
                     f"p={s['p_value']:.4f}, n={s['n_classes']}")

    # Step 4: O3 cross-check
    log.info("--- O3 cross-check ---")
    o3_check = o3_cross_check(df)
    log.info(f"O3 cross-check: rho={o3_check['rho']:.4f}, p={o3_check['p_value']:.4f}")

    # Step 5: Determine verdict
    threshold_persists = "false"
    if spearman["status"] == "significant":
        threshold_persists = "true"
    elif spearman["status"] == "suggestive" or sign["status"] == "vit_majority":
        threshold_persists = "suggestive"

    # Build results JSON
    results = {
        "spearman_o4": spearman,
        "sign_test_o4": sign,
        "sensitivity_analysis": sensitivity,
        "o3_cross_check": o3_check,
        "threshold_persists": threshold_persists,
        "interpretation": {
            "true": "Strong evidence: sample-efficiency threshold persists on O4 (rho > 0, p < 0.05)",
            "suggestive": "Suggestive but not significant: pattern visible but p >= 0.05",
            "false": "Threshold NOT confirmed on O4; report as O3-specific with caveat"
        }[threshold_persists],
        "backtracking_note": None if threshold_persists != "false" else
            "The sample-efficiency threshold may be O3-specific. Paper should report with caveat.",
        "n_classes_reliable": int(spearman["n_classes"]),
        "n_classes_total": int(len(df)),
    }

    # Add per-class breakdown for 100+ classes
    above_threshold = df[df["n_train_o3"] >= SAMPLE_EFFICIENCY_THRESHOLD].copy()
    results["above_threshold_classes"] = []
    for _, row in above_threshold.iterrows():
        results["above_threshold_classes"].append({
            "class": row["class"],
            "n_train_o3": int(row["n_train_o3"]),
            "f1_diff_o4": float(row["f1_diff_o4"]),
            "vit_wins": bool(row["f1_diff_o4"] > 0)
        })

    output_path = RESULTS_DIR / "o4_threshold_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved threshold test results: {output_path}")

    # Step 6: Figures
    log.info("--- Generating figures ---")
    make_threshold_scatter(df, spearman, FIGURES_DIR / "o4_threshold_scatter.png")
    make_degradation_chart(df, FIGURES_DIR / "o4_degradation_per_class.png")

    # Final summary
    log.info("=" * 60)
    log.info(f"VERDICT: threshold_persists = {threshold_persists}")
    log.info(f"  Spearman rho = {spearman['rho']:.4f}, p = {spearman['p_value']:.4f}")
    log.info(f"  Sign test: {sign['n_positive']}/{sign['n_total']} ViT wins for 100+ classes")
    log.info(f"  O3 cross-check: rho = {o3_check['rho']:.4f}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
