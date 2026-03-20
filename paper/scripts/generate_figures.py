#!/usr/bin/env python3
"""
Generate all journal-quality figures for the paper.
All data sourced from paper/data/paper_numbers.json or result CSVs.
No hardcoded numbers.

Figures:
  1. fig_per_class_f1.pdf     -- Grouped bar chart, all 23 classes
  2. fig_threshold_scatter.pdf -- N_train vs F1_diff scatter
  3. fig_confusion_matrices.pdf -- Side-by-side confusion matrices
  4. fig_o4_degradation.pdf   -- O3-to-O4 degradation per class
  5. fig_cw_veto.pdf           -- CW veto analysis summary
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(PROJECT_ROOT, "paper", "figures")

# CQG-compatible settings
SINGLE_COL = 3.4  # inches
DOUBLE_COL = 7.0  # inches

def setup_style():
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })

def load_paper_numbers():
    with open(os.path.join(PROJECT_ROOT, "paper", "data", "paper_numbers.json")) as f:
        return json.load(f)

def load_comparison_csv():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "results", "03-vit-rare-class", "comparison_table.csv"))

def load_o4_comparison_csv():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "results", "04-o4-validation", "o4_comparison_table.csv"))


def fig1_per_class_f1(pn):
    """Grouped bar chart of per-class F1 for all 23 classes."""
    pc = pn["per_class_o3"]
    classes = sorted(pc.keys(), key=lambda c: pc[c]["n_train"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    x = np.arange(len(classes))
    width = 0.35

    cnn_f1 = [pc[c]["cnn_f1"] for c in classes]
    vit_f1 = [pc[c]["vit_f1"] for c in classes]

    # Error bars from bootstrap CI
    cnn_err_lo = [pc[c]["cnn_f1"] - pc[c]["cnn_f1_ci_lower"] for c in classes]
    cnn_err_hi = [pc[c]["cnn_f1_ci_upper"] - pc[c]["cnn_f1"] for c in classes]
    vit_err_lo = [pc[c]["vit_f1"] - pc[c]["vit_f1_ci_lower"] for c in classes]
    vit_err_hi = [pc[c]["vit_f1_ci_upper"] - pc[c]["vit_f1"] for c in classes]

    bars_cnn = ax.bar(x - width/2, cnn_f1, width, label='CNN (ResNet-50v2)',
                      color='#4878CF', alpha=0.85,
                      yerr=[cnn_err_lo, cnn_err_hi], capsize=1.5, error_kw={'linewidth': 0.5})
    bars_vit = ax.bar(x + width/2, vit_f1, width, label='ViT-B/16',
                      color='#D65F5F', alpha=0.85,
                      yerr=[vit_err_lo, vit_err_hi], capsize=1.5, error_kw={'linewidth': 0.5})

    # Highlight rare classes with shading
    rare_indices = [i for i, c in enumerate(classes) if pc[c]["is_rare"]]
    for idx in rare_indices:
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.08, color='orange', zorder=0)

    # Add n_train as secondary labels
    ax2_labels = [str(pc[c]["n_train"]) for c in classes]

    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Glitch Class (sorted by training set size)')
    ax.set_xticks(x)
    display_names = [c.replace("_", "\n") for c in classes]
    ax.set_xticklabels(display_names, rotation=90, fontsize=6, ha='center')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_title('Per-class F1 scores on O3 test set (sorted by $N_\\mathrm{train}$)', fontsize=10)

    # Add text for rare class region
    if rare_indices:
        mid_rare = np.mean(rare_indices)
        ax.text(mid_rare, 1.02, 'Rare classes', ha='center', fontsize=7, color='darkorange', style='italic')

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "fig_per_class_f1.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig2_threshold_scatter(pn):
    """Scatter plot of N_train vs F1_diff for both O3 and O4."""
    pc_o3 = pn["per_class_o3"]
    pc_o4 = pn["per_class_o4"]
    threshold = pn["threshold_test"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))

    # O3 data
    classes_o3 = [c for c in pc_o3.keys()]
    n_train_o3 = [pc_o3[c]["n_train"] for c in classes_o3]
    f1_diff_o3 = [pc_o3[c]["f1_diff"] for c in classes_o3]

    # O4 data
    classes_o4 = [c for c in pc_o4.keys()]
    n_train_o4 = [pc_o4[c]["n_train_o3"] for c in classes_o4]
    f1_diff_o4 = [pc_o4[c]["f1_diff_o4"] for c in classes_o4]

    ax.scatter(n_train_o3, f1_diff_o3, s=25, alpha=0.7, color='#4878CF', marker='o', label='O3 test', zorder=3)
    ax.scatter(n_train_o4, f1_diff_o4, s=25, alpha=0.7, color='#D65F5F', marker='s', label='O4 test', zorder=3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel('$N_\\mathrm{train}$ (O3)')
    ax.set_ylabel('$\\Delta$F1 (ViT $-$ CNN)')

    # Annotate outliers
    label_classes = {"Power_Line", "Chirp", "Light_Modulation", "Paired_Doves"}
    for i, cls in enumerate(classes_o3):
        if cls in label_classes:
            offset = (5, 5)
            if cls == "Chirp":
                offset = (5, -10)
            ax.annotate(cls.replace("_", " "), (n_train_o3[i], f1_diff_o3[i]),
                       fontsize=5.5, textcoords='offset points', xytext=offset)

    # Add Spearman annotations
    rho_o3 = threshold["o3_spearman_rho"]
    p_o3 = threshold["o3_spearman_p"]
    rho_o4 = threshold["o4_spearman_rho"]
    p_o4 = threshold["o4_spearman_p"]

    text = (f"O3: $\\rho_s = {rho_o3:.3f}$, $p = {p_o3:.2f}$\n"
            f"O4: $\\rho_s = {rho_o4:.3f}$, $p = {p_o4:.2f}$")
    ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper left', fontsize=7)
    ax.set_title('Architecture preference vs.\\ training set size', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "fig_threshold_scatter.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig3_confusion_matrices(pn):
    """Side-by-side confusion matrices for ViT and CNN on O3."""
    # Load confusion matrices from numpy files
    cnn_cm = np.load(os.path.join(PROJECT_ROOT, "results", "03-vit-rare-class", "cnn_confusion_matrix.npy"))
    vit_cm = np.load(os.path.join(PROJECT_ROOT, "results", "03-vit-rare-class", "vit_confusion_matrix.npy"))

    # Load class names
    with open(os.path.join(PROJECT_ROOT, "results", "03-vit-rare-class", "class_names.json")) as f:
        class_names = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # Use log scale for visibility of rare classes
    cnn_cm_log = np.log10(cnn_cm + 1)
    vit_cm_log = np.log10(vit_cm + 1)
    vmax = max(cnn_cm_log.max(), vit_cm_log.max())

    short_names = [n[:8] for n in class_names]

    im1 = ax1.imshow(cnn_cm_log, cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
    ax1.set_title('CNN (ResNet-50v2)', fontsize=9)
    ax1.set_xlabel('Predicted', fontsize=8)
    ax1.set_ylabel('True', fontsize=8)
    ax1.set_xticks(range(len(short_names)))
    ax1.set_yticks(range(len(short_names)))
    ax1.set_xticklabels(short_names, rotation=90, fontsize=4)
    ax1.set_yticklabels(short_names, fontsize=4)

    im2 = ax2.imshow(vit_cm_log, cmap='Reds', aspect='auto', vmin=0, vmax=vmax)
    ax2.set_title('ViT-B/16', fontsize=9)
    ax2.set_xlabel('Predicted', fontsize=8)
    ax2.set_xticks(range(len(short_names)))
    ax2.set_yticks(range(len(short_names)))
    ax2.set_xticklabels(short_names, rotation=90, fontsize=4)
    ax2.set_yticklabels(short_names, fontsize=4)

    # Colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label='$\\log_{10}$(count + 1)')

    # Highlight rare classes on y-axis
    rare_classes = pn["dataset"]["rare_classes"]
    for ax in [ax1, ax2]:
        for i, name in enumerate(class_names):
            if name in rare_classes:
                ax.get_yticklabels()[i].set_color('darkorange')
                ax.get_yticklabels()[i].set_fontweight('bold')

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.15, top=0.92, wspace=0.25)
    outpath = os.path.join(FIGDIR, "fig_confusion_matrices.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig4_o4_degradation(pn):
    """Per-class F1 degradation from O3 to O4 for both models."""
    pc = pn["per_class_o4"]
    classes = sorted(pc.keys(), key=lambda c: pc[c]["n_train_o3"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))

    x = np.arange(len(classes))
    width = 0.35

    cnn_deg = [pc[c]["cnn_degradation"] for c in classes]
    vit_deg = [pc[c]["vit_degradation"] for c in classes]

    ax.bar(x - width/2, cnn_deg, width, label='CNN', color='#4878CF', alpha=0.85)
    ax.bar(x + width/2, vit_deg, width, label='ViT', color='#D65F5F', alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=-0.2, color='gray', linestyle='--', linewidth=0.8, label='$-20\\%$ threshold')
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.8)

    ax.set_ylabel('F1 change (O3 $\\to$ O4)')
    ax.set_xlabel('Glitch Class (sorted by $N_\\mathrm{train}$)')
    ax.set_xticks(x)
    display_names = [c.replace("_", "\n") for c in classes]
    ax.set_xticklabels(display_names, rotation=90, fontsize=5.5, ha='center')
    ax.legend(loc='lower left', fontsize=7)
    ax.set_title('Per-class generalization: O3 $\\to$ O4', fontsize=10)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "fig_o4_degradation.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def fig5_cw_veto(pn):
    """CW veto analysis summary: matched-deadtime comparison + per-class duty cycle."""
    cw = pn["cw"]
    cw_pc = pn["cw_per_class"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8),
                                    gridspec_kw={'width_ratios': [1, 2]})

    # Left panel: matched-deadtime bar chart
    models = ['CNN', 'ViT']
    effs = [cw["efficiency_at_matched_cnn"], cw["efficiency_at_matched_vit"]]
    colors = ['#4878CF', '#D65F5F']
    bars = ax1.bar(models, effs, color=colors, alpha=0.85, width=0.5)
    ax1.set_ylabel('Veto Efficiency')
    ax1.set_ylim(0.7, 0.76)
    ax1.set_title(f'Matched deadtime\n({cw["matched_deadtime"]*100:.1f}%)', fontsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, effs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Right panel: per-class duty cycle comparison
    cw_classes = ["Low_Frequency_Lines", "Scattered_Light", "Power_Line",
                  "Violin_Mode", "1080Lines", "Whistle", "Low_Frequency_Burst"]

    x = np.arange(len(cw_classes))
    width = 0.35

    vit_dc = [cw_pc[c]["vit_dc"] for c in cw_classes]
    cnn_dc = [cw_pc[c]["cnn_dc"] for c in cw_classes]

    ax2.bar(x - width/2, cnn_dc, width, label='CNN', color='#4878CF', alpha=0.85)
    ax2.bar(x + width/2, vit_dc, width, label='ViT', color='#D65F5F', alpha=0.85)

    display = [c.replace("_", "\n") for c in cw_classes]
    ax2.set_xticks(x)
    ax2.set_xticklabels(display, rotation=90, fontsize=6)
    ax2.set_ylabel('Duty Cycle')
    ax2.set_ylim(0.85, 1.005)
    ax2.legend(loc='lower left', fontsize=7)
    ax2.set_title('Per-class duty cycle\n(CW-critical classes, O4)', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(FIGDIR, "fig_cw_veto.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def main():
    setup_style()
    os.makedirs(FIGDIR, exist_ok=True)
    pn = load_paper_numbers()

    print("Generating figures...")
    fig1_per_class_f1(pn)
    fig2_threshold_scatter(pn)
    fig3_confusion_matrices(pn)
    fig4_o4_degradation(pn)
    fig5_cw_veto(pn)
    print("All figures generated.")

if __name__ == "__main__":
    main()
