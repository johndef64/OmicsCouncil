"""Composite results figure for the OmicsCouncil medium-weight paper.

Generates a 4-panel figure (PDF + PNG) saved to ``paper/figures/``:

  (A) 5-fold CV accuracy + macro-F1 bar chart, 4 models.
  (B) 5-fold CV ECE bar chart, 4 models — lower is better.
  (C) K ablation: accuracy and faithfulness vs top_features_per_agent.
  (D) Marker recovery: per-trace count of canonical PAM50 markers.

Inputs are read from ``results/`` (produced by run_classification_cv.py and
run_classification.py). Panel (D) re-runs the pipeline once to extract the
per-trace marker count histogram (cheap: ~30s on cached pipeline).

Usage:
    python src/plots.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Make the package importable for panel (D).
sys.path.insert(0, str(ROOT / "src"))

# Consistent palette across panels.
PALETTE = {
    "Concat+LogReg": "#7eb3d6",
    "Concat+MLP":    "#a3c47a",
    "Late-fusion":   "#e1a85b",
    "OmicsCouncil":  "#c0504d",
}
MODELS_ORDER = ["Concat+LogReg", "Concat+MLP", "Late-fusion", "OmicsCouncil"]

# Nadeau-Bengio corrected paired test of each baseline's ECE against
# OmicsCouncil's, over the 100 folds of the 20x5 protocol.
# Produced by experiments/run_significance.py.
ECE_PVALUES = {
    "Concat+LogReg": "n.s.",
    "Concat+MLP":    "p=0.0001",
    "Late-fusion":   "n.s.",
}


# --------------------------------------------------------------------------- #
# Data readers
# --------------------------------------------------------------------------- #
def load_cv_summary(path: Path) -> Dict[str, Dict[str, float]]:
    """summary[model] -> {metric_mean, metric_std} for the 5-fold CV file."""
    blob = json.loads(path.read_text())
    out = {}
    for row in blob["summary"]:
        out[row["model"]] = row
    return out


def load_k_ablation() -> List[Tuple[int, dict]]:
    """Read K=40 / 60 / 80 CV files. Returns [(K, summary_for_OC), ...]."""
    files = [(40, "tcga_brca_cv20x5.json"),
             (60, "tcga_brca_cv20x5_k60.json"),
             (80, "tcga_brca_cv20x5_k80.json")]
    out = []
    for k, fname in files:
        blob = json.loads((RESULTS_DIR / fname).read_text())
        oc = next(r for r in blob["summary"] if r["model"] == "OmicsCouncil")
        oc["faithfulness"] = blob["trustworthy"].get("faithfulness_mean", float("nan"))
        oc["containment"] = blob["trustworthy"].get("halluc_containment_rate", float("nan"))
        out.append((k, oc))
    return out


def compute_marker_distribution() -> Dict[int, int]:
    """Re-run the K=40 pipeline once and return a histogram of canonical-marker
    counts per verified trace."""
    from omicscouncil.config import load_config
    from omicscouncil.data import build_multimodal_dataset, train_test_split_dataset
    from omicscouncil.pipeline import OmicsCouncilPipeline
    cfg = load_config(str(ROOT / "configs" / "tcga_brca.yaml"))
    ds = build_multimodal_dataset(cfg)
    train, test = train_test_split_dataset(ds, cfg.test_size, cfg.seed)
    pipe = OmicsCouncilPipeline(cfg).fit(train)
    _, traces = pipe.transform(test)
    markers = {m.upper() for m in cfg.reference_markers}
    counts = Counter()
    for tr in traces:
        symbols = {c.subject.upper() for c in tr.verified_claims}
        counts[len(symbols & markers)] += 1
    return dict(counts)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })


def panel_accuracy(ax, summary):
    """Side-by-side bars: accuracy and macro-F1 for the four models."""
    x = np.arange(len(MODELS_ORDER))
    w = 0.38
    acc_means = [summary[m]["accuracy_mean"] for m in MODELS_ORDER]
    acc_stds = [summary[m]["accuracy_std"] for m in MODELS_ORDER]
    f1_means = [summary[m]["macro_f1_mean"] for m in MODELS_ORDER]
    f1_stds = [summary[m]["macro_f1_std"] for m in MODELS_ORDER]

    bars_a = ax.bar(x - w/2, acc_means, w, yerr=acc_stds,
                    color=[PALETTE[m] for m in MODELS_ORDER],
                    edgecolor="black", linewidth=0.5, capsize=2.5, label="Accuracy")
    bars_b = ax.bar(x + w/2, f1_means, w, yerr=f1_stds,
                    color=[PALETTE[m] for m in MODELS_ORDER],
                    edgecolor="black", linewidth=0.5, hatch="///",
                    alpha=0.55, capsize=2.5, label="Macro-F1")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_ORDER, rotation=20, ha="right")
    ax.set_ylim(0.6, 0.99)
    ax.set_ylabel("Score (20x5 repeated CV)")
    ax.set_title("(A) Classification: accuracy / macro-F1")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              ncol=2, fontsize=7,
              handles=[plt.Rectangle((0,0),1,1, color="0.5", label="Accuracy"),
                       plt.Rectangle((0,0),1,1, color="0.5", hatch="///",
                                     alpha=0.55, label="Macro-F1")])


def panel_ece(ax, summary):
    """ECE bar chart with explicit 'lower is better' marker."""
    x = np.arange(len(MODELS_ORDER))
    means = [summary[m]["ece_mean"] for m in MODELS_ORDER]
    stds = [summary[m]["ece_std"] for m in MODELS_ORDER]
    ax.bar(x, means, 0.6, yerr=stds,
           color=[PALETTE[m] for m in MODELS_ORDER],
           edgecolor="black", linewidth=0.5, capsize=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Expected Calibration Error  ↓")
    ax.set_title("(B) Calibration (ECE; lower is better)")
    ax.set_ylim(0, max(means) * 1.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Highlight the OC bar with a small annotation.
    oc_idx = MODELS_ORDER.index("OmicsCouncil")
    ax.annotate("best", xy=(oc_idx, means[oc_idx]),
                xytext=(oc_idx, means[oc_idx] + 0.025),
                ha="center", fontsize=8, color="black",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    # Paired per-fold test of each baseline against OmicsCouncil, so the
    # reader sees the significance of the *differences* and not only the
    # overlap of the per-model bars. See run_significance.py.
    for m, label in ECE_PVALUES.items():
        i = MODELS_ORDER.index(m)
        ax.text(i, means[i] + stds[i] + 0.012, label,
                ha="center", fontsize=7, color="0.25")


def panel_k_ablation(ax, k_data):
    """K vs accuracy and K vs faithfulness on twin y-axes."""
    Ks = [k for k, _ in k_data]
    accs = [d["accuracy_mean"] for _, d in k_data]
    accs_std = [d["accuracy_std"] for _, d in k_data]
    f1s = [d["macro_f1_mean"] for _, d in k_data]
    faiths = [d["faithfulness"] for _, d in k_data]

    line_a = ax.errorbar(Ks, accs, yerr=accs_std, marker="o", color="#c0504d",
                         linewidth=1.5, capsize=2.5, label="Accuracy")
    line_f1 = ax.plot(Ks, f1s, marker="s", color="#7eb3d6", linewidth=1.5,
                      linestyle="--", label="Macro-F1")
    ax.set_xlabel("top_features_per_agent (K)")
    ax.set_ylabel("Accuracy / Macro-F1", color="#c0504d")
    ax.set_xticks(Ks)
    # Wide enough to show every error bar in full, caps included: a tighter
    # range clips them and makes the intervals look different in width than
    # they are.
    ax.set_ylim(0.75, 0.90)
    ax.tick_params(axis="y", colors="#c0504d")
    ax.set_title("(C) Ablation: K vs accuracy vs faithfulness")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    ax2 = ax.twinx()
    line_faith = ax2.plot(Ks, faiths, marker="^", color="#5a8a3a",
                          linewidth=1.5, label="Faithfulness")
    ax2.set_ylabel("Faithfulness  ↑", color="#5a8a3a")
    ax2.set_ylim(0.0, 0.06)
    ax2.tick_params(axis="y", colors="#5a8a3a")
    ax2.spines["top"].set_visible(False)

    # Star K=40 as chosen.
    ax.axvline(40, color="grey", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.text(40, 0.888, "chosen", ha="center", fontsize=8, color="grey")

    # Combined legend.
    handles = [line_a, line_f1[0], line_faith[0]]
    labels = ["Accuracy", "Macro-F1", "Faithfulness"]
    ax.legend(handles, labels, loc="lower right", frameon=False)


def panel_marker_recovery(ax, marker_counts):
    """Histogram of number of PAM50 canonical markers per verified trace."""
    max_k = max(marker_counts.keys())
    xs = list(range(0, max_k + 1))
    ys = [marker_counts.get(k, 0) for k in xs]
    total = sum(ys)
    pct = [100 * y / total for y in ys]

    bars = ax.bar(xs, pct, 0.7, color="#c0504d", edgecolor="black",
                  linewidth=0.5, alpha=0.85)
    for x, p in zip(xs, pct):
        if p >= 1:
            ax.text(x, p + 1.5, f"{p:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xlabel("Canonical PAM50 markers per verified trace")
    ax.set_ylabel("Test traces (%)")
    ax.set_title("(D) Marker recovery (test split, K=40)")
    ax.set_ylim(0, max(pct) * 1.18)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    _set_paper_style()

    cv = load_cv_summary(RESULTS_DIR / "tcga_brca_cv20x5.json")
    k_data = load_k_ablation()
    print("Computing marker recovery histogram (this re-runs the K=40 pipeline once)...")
    marker_counts = compute_marker_distribution()
    print(f"  marker count distribution: {sorted(marker_counts.items())}")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    panel_accuracy(axes[0, 0], cv)
    panel_ece(axes[0, 1], cv)
    panel_k_ablation(axes[1, 0], k_data)
    panel_marker_recovery(axes[1, 1], marker_counts)
    fig.tight_layout()

    pdf = FIG_DIR / "results_composite.pdf"
    png = FIG_DIR / "results_composite.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
