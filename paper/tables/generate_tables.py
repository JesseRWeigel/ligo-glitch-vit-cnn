#!/usr/bin/env python3
"""
Generate LaTeX table source files from paper_numbers.json.
All numbers come from the extraction pipeline -- no hardcoded values.
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_paper_numbers():
    path = os.path.join(PROJECT_ROOT, "paper", "data", "paper_numbers.json")
    with open(path) as f:
        return json.load(f)

def fmt(val, decimals=3):
    """Format a float to given decimal places."""
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"

def fmt_ci(val, lo, hi, decimals=3):
    """Format value [CI_lo, CI_hi]."""
    return f"{fmt(val, decimals)} [{fmt(lo, decimals)}, {fmt(hi, decimals)}]"

def generate_table1(pn, outdir):
    """Overall comparison table: O3 and O4 metrics."""
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall classification performance. Macro-F1 (primary metric) with 95\% bootstrap CIs (10\,000 resamples). $p$-values from paired percentile bootstrap (one-sided: ViT $>$ CNN).}")
    lines.append(r"\label{tab:overall}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline\hline")
    lines.append(r"Dataset & Metric & CNN & ViT & $p$-value \\")
    lines.append(r"\hline")

    # O3 overall
    cnn_o3 = fmt_ci(pn["o3_cnn"]["macro_f1"], pn["o3_cnn"]["macro_f1_ci_lower"], pn["o3_cnn"]["macro_f1_ci_upper"])
    vit_o3 = fmt_ci(pn["o3_vit"]["macro_f1"], pn["o3_vit"]["macro_f1_ci_lower"], pn["o3_vit"]["macro_f1_ci_upper"])
    lines.append(f"O3 & Macro-F1 (all) & {cnn_o3} & {vit_o3} & {pn['paired_bootstrap']['overall_p_value']} \\\\")

    # O3 rare
    cnn_rare = fmt_ci(pn["o3_cnn"]["rare_class_macro_f1"], pn["o3_cnn"]["rare_class_macro_f1_ci_lower"], pn["o3_cnn"]["rare_class_macro_f1_ci_upper"])
    vit_rare = fmt_ci(pn["o3_vit"]["rare_class_macro_f1"], pn["o3_vit"]["rare_class_macro_f1_ci_lower"], pn["o3_vit"]["rare_class_macro_f1_ci_upper"])
    lines.append(f"O3 & Macro-F1 (rare) & {cnn_rare} & {vit_rare} & {pn['paired_bootstrap']['rare_class_p_value']} \\\\")

    # O4 overall
    cnn_o4 = fmt_ci(pn["o4_cnn"]["macro_f1"], pn["o4_cnn"]["macro_f1_ci_lower"], pn["o4_cnn"]["macro_f1_ci_upper"])
    vit_o4 = fmt_ci(pn["o4_vit"]["macro_f1"], pn["o4_vit"]["macro_f1_ci_lower"], pn["o4_vit"]["macro_f1_ci_upper"])
    lines.append(f"O4 & Macro-F1 (all) & {cnn_o4} & {vit_o4} & --- \\\\")

    # Degradation
    cnn_deg = f"{pn['o4_cnn']['degradation_relative']*100:.1f}\\%"
    vit_deg = f"{pn['o4_vit']['degradation_relative']*100:.1f}\\%"
    lines.append(f"O3$\\to$O4 & Degradation & {cnn_deg} & {vit_deg} & --- \\\\")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(os.path.join(outdir, "table_overall.tex"), "w") as f:
        f.write("\n".join(lines))


def generate_table2(pn, outdir):
    """Full per-class F1 table for all 23 classes on O3."""
    pc = pn["per_class_o3"]
    # Sort by training set size
    classes_sorted = sorted(pc.keys(), key=lambda c: pc[c]["n_train"])

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class F1 scores on the O3 test set for all 23 classes, sorted by training set size. Bold indicates the better model per class. Rare classes ($N_\mathrm{train} < 200$) are marked with $\dagger$.}")
    lines.append(r"\label{tab:perclass}")
    lines.append(r"\begin{tabular}{lrrrrrl}")
    lines.append(r"\hline\hline")
    lines.append(r"Class & $N_\mathrm{train}$ & $N_\mathrm{test}$ & CNN F1 & ViT F1 & $\Delta$F1 & Winner \\")
    lines.append(r"\hline")

    for cls in classes_sorted:
        d = pc[cls]
        rare_mark = "$^\\dagger$" if d["is_rare"] else ""
        cnn_f1 = d["cnn_f1"]
        vit_f1 = d["vit_f1"]
        diff = d["f1_diff"]

        cnn_str = fmt(cnn_f1)
        vit_str = fmt(vit_f1)

        if vit_f1 > cnn_f1:
            vit_str = f"\\textbf{{{vit_str}}}"
            winner = "ViT"
        elif cnn_f1 > vit_f1:
            cnn_str = f"\\textbf{{{cnn_str}}}"
            winner = "CNN"
        else:
            winner = "Tie"

        cls_display = cls.replace("_", r"\_")
        lines.append(f"{cls_display}{rare_mark} & {d['n_train']} & {d['n_test']} & {cnn_str} & {vit_str} & {fmt(diff, 3)} & {winner} \\\\")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(os.path.join(outdir, "table_perclass.tex"), "w") as f:
        f.write("\n".join(lines))


def generate_table3(pn, outdir):
    """CW-critical class analysis table."""
    cw = pn["cw_per_class"]
    classes = ["Low_Frequency_Lines", "Scattered_Light", "Violin_Mode",
               "Power_Line", "1080Lines", "Low_Frequency_Burst", "Whistle"]

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering")
    lines.append(r"\caption{CW-critical glitch class analysis on O4 data. Duty cycle (DC) is the fraction of observation time retained after vetoing glitches of each class. $\Delta_\mathrm{DC} = \mathrm{DC}_\mathrm{ViT} - \mathrm{DC}_\mathrm{CNN}$; negative values indicate CNN retains more livetime.}")
    lines.append(r"\label{tab:cw}")
    lines.append(r"\begin{tabular}{llrrrrl}")
    lines.append(r"\hline\hline")
    lines.append(r"Class & CW impact & $N_\mathrm{O4}$ & ViT DC & CNN DC & $\Delta_\mathrm{DC}$ & Advantage \\")
    lines.append(r"\hline")

    for cls in classes:
        d = cw[cls]
        cls_display = cls.replace("_", r"\_")
        adv = "ViT" if d["delta_dc"] < 0 and d["f1_diff_o4"] > 0 else ("CNN" if d["f1_diff_o4"] < 0 else "---")
        # Actually use the simple criterion: which model has higher F1 on O4 for this class
        if d["f1_diff_o4"] > 0.01:
            adv = "ViT"
        elif d["f1_diff_o4"] < -0.01:
            adv = "CNN"
        else:
            adv = "$\\approx$"

        ci_str = f"[{fmt(d['delta_dc_ci'][0], 4)}, {fmt(d['delta_dc_ci'][1], 4)}]"
        lines.append(f"{cls_display} & {d['cw_impact']} & {d['n_true_o4']} & {fmt(d['vit_dc'], 3)} & {fmt(d['cnn_dc'], 3)} & {fmt(d['delta_dc'], 4)} {ci_str} & {adv} \\\\")

    lines.append(r"\hline")
    # Combined row
    cw_overall = pn["cw"]
    ci_str = f"[{fmt(cw_overall['overall_delta_dc_ci_lower'], 3)}, {fmt(cw_overall['overall_delta_dc_ci_upper'], 3)}]"
    lines.append(f"\\textbf{{Combined}} & ALL & --- & {fmt(cw_overall['overall_duty_cycle_vit'], 3)} & {fmt(cw_overall['overall_duty_cycle_cnn'], 3)} & {fmt(cw_overall['overall_delta_dc'], 3)} {ci_str} & CNN \\\\")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(os.path.join(outdir, "table_cw.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    pn = load_paper_numbers()
    outdir = os.path.join(PROJECT_ROOT, "paper", "tables")
    os.makedirs(outdir, exist_ok=True)

    generate_table1(pn, outdir)
    generate_table2(pn, outdir)
    generate_table3(pn, outdir)

    print("Generated: table_overall.tex, table_perclass.tex, table_cw.tex")

if __name__ == "__main__":
    main()
