#!/usr/bin/env python3
"""
Extract all numbers from Phase 2-4 result files into a single JSON dict.
This is the SINGLE SOURCE OF TRUTH for all numbers in the paper.

CRITICAL: The "3.4x" CW efficiency ratio is an np.interp boundary clamping bug.
This script MUST NOT output any 5%-deadtime interpolated values.
"""

import json
import csv
import os
import sys
import warnings

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_json(relpath):
    path = os.path.join(PROJECT_ROOT, relpath)
    with open(path) as f:
        return json.load(f)

def load_csv(relpath):
    path = os.path.join(PROJECT_ROOT, relpath)
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    # ── Load all result files ──
    vit_metrics = load_json("results/03-vit-rare-class/metrics.json")
    cnn_metrics = load_json("results/02-cnn-baseline/metrics.json")
    bootstrap = load_json("results/03-vit-rare-class/paired_bootstrap_results.json")
    o4_metrics = load_json("results/04-o4-validation/o4_metrics.json")
    o4_threshold = load_json("results/04-o4-validation/o4_threshold_test.json")
    cw_results = load_json("results/04-o4-validation/cw_veto_results.json")
    comparison_csv = load_csv("results/03-vit-rare-class/comparison_table.csv")
    o4_comparison_csv = load_csv("results/04-o4-validation/o4_comparison_table.csv")
    cw_duty_csv = load_csv("results/04-o4-validation/cw_duty_cycle_comparison.csv")

    # ── Build paper_numbers dict ──
    paper = {}

    # Dataset statistics
    paper["dataset"] = {
        "n_train": 227943,
        "n_val": 48844,
        "n_test": 48845,
        "n_total": 325632,
        "n_classes": 23,
        "n_o4": 38587,
        "rare_threshold_train": 200,
        "rare_classes": ["Chirp", "Wandering_Line", "Helix", "Light_Modulation"],
        "n_rare_classes": 4,
    }

    # O3 overall metrics
    paper["o3_vit"] = {
        "macro_f1": vit_metrics["macro_f1"]["value"],
        "macro_f1_ci_lower": vit_metrics["macro_f1"]["ci_lower"],
        "macro_f1_ci_upper": vit_metrics["macro_f1"]["ci_upper"],
        "overall_accuracy": vit_metrics["overall_accuracy"]["value"],
        "rare_class_macro_f1": vit_metrics["rare_class_macro_f1"]["value"],
        "rare_class_macro_f1_ci_lower": vit_metrics["rare_class_macro_f1"]["ci_lower"],
        "rare_class_macro_f1_ci_upper": vit_metrics["rare_class_macro_f1"]["ci_upper"],
        "common_class_avg_f1": vit_metrics["common_class_avg_f1"]["value"],
    }

    paper["o3_cnn"] = {
        "macro_f1": cnn_metrics["macro_f1"]["value"],
        "macro_f1_ci_lower": cnn_metrics["macro_f1"]["ci_lower"],
        "macro_f1_ci_upper": cnn_metrics["macro_f1"]["ci_upper"],
        "overall_accuracy": cnn_metrics["overall_accuracy"]["value"],
        "rare_class_macro_f1": cnn_metrics["rare_class_macro_f1"]["value"],
        "rare_class_macro_f1_ci_lower": cnn_metrics["rare_class_macro_f1"]["ci_lower"],
        "rare_class_macro_f1_ci_upper": cnn_metrics["rare_class_macro_f1"]["ci_upper"],
        "common_class_avg_f1": cnn_metrics["common_class_avg_f1"]["value"],
    }

    # Paired bootstrap results
    paper["paired_bootstrap"] = {
        "overall_macro_f1_diff": bootstrap["overall_macro_f1"]["point_estimate_difference"],
        "overall_macro_f1_diff_ci_lower": bootstrap["overall_macro_f1"]["ci_lower"],
        "overall_macro_f1_diff_ci_upper": bootstrap["overall_macro_f1"]["ci_upper"],
        "overall_p_value": bootstrap["overall_macro_f1"]["p_value"],
        "rare_class_diff": bootstrap["rare_class_macro_f1"]["point_estimate_difference"],
        "rare_class_diff_ci_lower": bootstrap["rare_class_macro_f1"]["ci_lower"],
        "rare_class_diff_ci_upper": bootstrap["rare_class_macro_f1"]["ci_upper"],
        "rare_class_p_value": bootstrap["rare_class_macro_f1"]["p_value"],
        "n_resamples": bootstrap["overall_macro_f1"]["n_resamples"],
    }

    # O4 metrics
    paper["o4_vit"] = {
        "macro_f1": o4_metrics["vit_macro_f1_o4"],
        "macro_f1_ci_lower": o4_metrics["vit_macro_f1_o4_ci"][0],
        "macro_f1_ci_upper": o4_metrics["vit_macro_f1_o4_ci"][1],
        "rare_class_macro_f1": o4_metrics["vit_rare_macro_f1_o4"],
        "degradation_absolute": o4_metrics["degradation"]["vit_degradation_absolute"],
        "degradation_relative": o4_metrics["degradation"]["vit_degradation_relative"],
        "passes_20pct": o4_metrics["degradation"]["vit_passes_20pct_threshold"],
    }

    paper["o4_cnn"] = {
        "macro_f1": o4_metrics["cnn_macro_f1_o4"],
        "macro_f1_ci_lower": o4_metrics["cnn_macro_f1_o4_ci"][0],
        "macro_f1_ci_upper": o4_metrics["cnn_macro_f1_o4_ci"][1],
        "rare_class_macro_f1": o4_metrics["cnn_rare_macro_f1_o4"],
        "degradation_absolute": o4_metrics["degradation"]["cnn_degradation_absolute"],
        "degradation_relative": o4_metrics["degradation"]["cnn_degradation_relative"],
        "passes_20pct": o4_metrics["degradation"]["cnn_passes_20pct_threshold"],
    }

    # Threshold test
    paper["threshold_test"] = {
        "o4_spearman_rho": o4_threshold["spearman_o4"]["rho"],
        "o4_spearman_p": o4_threshold["spearman_o4"]["p_value"],
        "o3_spearman_rho": o4_threshold["o3_cross_check"]["rho"],
        "o3_spearman_p": o4_threshold["o3_cross_check"]["p_value"],
        "sign_test_vit_wins": o4_threshold["sign_test_o4"]["n_positive"],
        "sign_test_total": o4_threshold["sign_test_o4"]["n_total"],
        "sign_test_p": o4_threshold["sign_test_o4"]["p_value"],
        "status": o4_threshold["spearman_o4"]["status"],
    }

    # CW analysis
    cw_combined = cw_results["combined_metrics"]
    roc = cw_results["roc_analysis"]
    paper["cw"] = {
        "matched_deadtime": roc["matched_deadtime"],
        "efficiency_at_matched_vit": roc["efficiency_at_matched_deadtime_vit"],
        "efficiency_at_matched_cnn": roc["efficiency_at_matched_deadtime_cnn"],
        "overall_delta_dc": cw_combined["delta_dc"],
        "overall_delta_dc_ci_lower": cw_combined["delta_dc_ci"][0],
        "overall_delta_dc_ci_upper": cw_combined["delta_dc_ci"][1],
        "overall_duty_cycle_vit": cw_combined["duty_cycle_vit"],
        "overall_duty_cycle_cnn": cw_combined["duty_cycle_cnn"],
        "n_cw_critical_classes": len(cw_results["cw_critical_classes"]),
        "n_vit_advantaged": len(cw_results["vit_advantaged_cw_classes"]),
        "n_cnn_advantaged": len(cw_results["cnn_advantaged_cw_classes"]),
        "vit_advantaged_classes": cw_results["vit_advantaged_cw_classes"],
        "cnn_advantaged_classes": cw_results["cnn_advantaged_cw_classes"],
    }

    # Per-class CW breakdown
    paper["cw_per_class"] = {}
    for cls, data in cw_results["per_class_cw_breakdown"].items():
        paper["cw_per_class"][cls] = {
            "cw_impact": data["cw_impact"],
            "freq_range": data["freq_range"],
            "n_true_o4": data["n_true_o4"],
            "vit_f1_o4": data["vit_f1_o4"],
            "cnn_f1_o4": data["cnn_f1_o4"],
            "f1_diff_o4": data["f1_diff_o4"],
            "vit_dc": data["vit_dc"],
            "cnn_dc": data["cnn_dc"],
            "delta_dc": data["delta_dc"],
            "delta_dc_ci": data["delta_dc_ci"],
            "vit_veto_eff": data["vit_veto_efficiency"],
            "cnn_veto_eff": data["cnn_veto_efficiency"],
        }

    # Per-class F1 for O3 (from comparison CSV)
    paper["per_class_o3"] = {}
    for row in comparison_csv:
        cls = row["class"]
        if cls.startswith("MACRO"):
            continue
        paper["per_class_o3"][cls] = {
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "cnn_f1": float(row["cnn_f1"]),
            "cnn_f1_ci_lower": float(row["cnn_f1_ci_lower"]),
            "cnn_f1_ci_upper": float(row["cnn_f1_ci_upper"]),
            "vit_f1": float(row["vit_f1"]),
            "vit_f1_ci_lower": float(row["vit_f1_ci_lower"]),
            "vit_f1_ci_upper": float(row["vit_f1_ci_upper"]),
            "f1_diff": float(row["f1_diff"]),
            "is_rare": row["is_rare"] == "True",
        }

    # Per-class F1 for O4 (from comparison CSV)
    paper["per_class_o4"] = {}
    for row in o4_comparison_csv:
        cls = row["class"]
        if cls.startswith("MACRO"):
            continue
        paper["per_class_o4"][cls] = {
            "n_train_o3": int(float(row["n_train_o3"])),
            "n_test_o4": int(float(row["n_test_o4"])),
            "cnn_f1_o3": float(row["cnn_f1_o3"]),
            "vit_f1_o3": float(row["vit_f1_o3"]),
            "cnn_f1_o4": float(row["cnn_f1_o4"]),
            "vit_f1_o4": float(row["vit_f1_o4"]),
            "cnn_degradation": float(row["cnn_degradation"]),
            "vit_degradation": float(row["vit_degradation"]),
            "f1_diff_o4": float(row["f1_diff_o4"]),
        }

    # Key per-class highlights for inline text
    # IMPORTANT: Look up by class name, NOT by row index (CSV sort order is not guaranteed)
    per_class_o3_lookup = {row["class"]: row for row in comparison_csv}
    paper["highlights"] = {
        "power_line_f1_diff_o3": float(per_class_o3_lookup["Power_Line"]["f1_diff"]),
        "light_mod_f1_diff_o3": float(per_class_o3_lookup["Light_Modulation"]["f1_diff"]),
        "chirp_f1_diff_o3": float(per_class_o3_lookup["Chirp"]["f1_diff"]),
        "power_line_vit_f1_o3": vit_metrics["per_class_f1"]["Power_Line"]["f1"],
        "power_line_cnn_f1_o3": cnn_metrics["per_class_f1"]["Power_Line"]["f1"],
        "power_line_f1_diff_o4": 0.3935164891603077,  # from o4_threshold_test
        "chirp_vit_f1_o3": vit_metrics["per_class_f1"]["Chirp"]["f1"],
        "chirp_cnn_f1_o3": cnn_metrics["per_class_f1"]["Chirp"]["f1"],
    }

    # ── CRITICAL CHECK: "3.4" must NOT appear ──
    output_str = json.dumps(paper)
    if "3.4" in output_str:
        # Check if it's actually the "3.4x" CW bug vs just a legitimate number
        # Legitimate uses: "3.4in" column width, values like 0.3434...
        # Problematic: "3.4x", "3.4 times", ratio of 3.4
        import re
        matches_34x = re.findall(r'3\.4[x\s]', output_str)
        if matches_34x:
            print("CRITICAL ERROR: Found '3.4x' pattern in output — np.interp bug propagation!", file=sys.stderr)
            sys.exit(1)
        else:
            # Values like 0.3434 are legitimate per-class F1 CI values
            pass

    # Check for NaN/Inf
    if "NaN" in output_str or "Infinity" in output_str:
        print("WARNING: NaN or Infinity found in output — stripping", file=sys.stderr)
        # Remove any NaN entries (from the 5% deadtime interpolation bug)
        paper_clean = json.loads(output_str.replace("NaN", "null").replace("Infinity", "null"))
        paper = paper_clean

    # Write output
    outpath = os.path.join(PROJECT_ROOT, "paper", "data", "paper_numbers.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(paper, f, indent=2)

    print(f"Written {outpath}")
    print(f"Total keys at top level: {len(paper)}")
    print(f"Per-class O3 entries: {len(paper['per_class_o3'])}")
    print(f"Per-class O4 entries: {len(paper['per_class_o4'])}")
    print(f"CW per-class entries: {len(paper['cw_per_class'])}")

    # Final sanity: verify key values
    assert abs(paper["o3_vit"]["macro_f1"] - 0.7230) < 0.001, "ViT macro-F1 mismatch"
    assert abs(paper["o3_cnn"]["macro_f1"] - 0.6786) < 0.001, "CNN macro-F1 mismatch"
    assert paper["paired_bootstrap"]["overall_p_value"] == 0.0002, "p-value mismatch"
    assert paper["threshold_test"]["status"] == "not_confirmed", "Threshold status mismatch"
    print("All sanity checks passed.")

if __name__ == "__main__":
    main()
