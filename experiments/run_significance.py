"""Reproduces every statistical number reported in the paper.

Reads the JSON dumps already produced by ``run_classification_cv_repeated.py``
-- no model is refitted -- and reports:

  * OmicsCouncil vs each baseline, per metric, on the 100 folds of the
    20x5 protocol (Sec. "Significance of the differences");
  * OmicsCouncil at K=60 / K=80 against the chosen K=40, paired on the same
    folds (Sec. "Ablation on K").

Both use the Nadeau-Bengio corrected resampled t-test, which inflates the
variance of the paired differences to account for the overlap between the
training sets of cross-validation folds.  The uncorrected paired t-test is
anti-conservative here and is printed only for contrast.

    python experiments/run_significance.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS = os.path.join(_ROOT, "results")

BASELINES = ["Concat+LogReg", "Concat+MLP", "Late-fusion"]
METRICS = ["ece", "accuracy", "macro_f1", "balanced_accuracy"]
LOWER_IS_BETTER = {"ece"}
K_FILES = {40: "tcga_brca_cv20x5.json",
           60: "tcga_brca_cv20x5_k60.json",
           80: "tcga_brca_cv20x5_k80.json"}


def per_fold(path: str, model: str, metric: str) -> Tuple[List, np.ndarray]:
    """Metric values keyed by (repeat, fold) so runs can be paired exactly."""
    blob = json.load(open(path, encoding="utf-8"))
    rows = {(r["repeat"], r["fold"]): r[metric]
            for r in blob["per_fold"] if r["model"] == model}
    keys = sorted(rows)
    return keys, np.array([rows[k] for k in keys], dtype=float)


def nadeau_bengio(d: np.ndarray, n_folds: int = 5) -> float:
    """Corrected resampled t-test p-value on paired differences `d`."""
    corr = 1.0 / len(d) + 1.0 / (n_folds - 1)
    t = d.mean() / np.sqrt(corr * d.var(ddof=1))
    return float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def report(d: np.ndarray, label: str) -> Dict[str, float]:
    out = {
        "n": len(d),
        "wins": int((d > 0).sum()),
        "mean_diff": float(d.mean()),
        "p_naive": float(stats.ttest_1samp(d, 0.0).pvalue),
        "p_nadeau_bengio": nadeau_bengio(d),
    }
    print(f"  {label:34s} {out['wins']:>3}/{out['n']:<4} "
          f"d={out['mean_diff']:+.4f}  "
          f"p_NB={out['p_nadeau_bengio']:.4f}   "
          f"(uncorrected {out['p_naive']:.4f})")
    return out


def main() -> None:
    payload: Dict[str, object] = {"protocol": "20x5 repeated stratified CV"}
    main_file = os.path.join(RESULTS, K_FILES[40])

    print("=" * 78)
    print("OmicsCouncil vs baselines -- paired over 100 folds")
    print("positive mean_diff = OmicsCouncil better")
    print("=" * 78)
    tests: Dict[str, Dict[str, Dict]] = {}
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        tests[metric] = {}
        _, oc = per_fold(main_file, "OmicsCouncil", metric)
        for b in BASELINES:
            _, base = per_fold(main_file, b, metric)
            d = (base - oc) if metric in LOWER_IS_BETTER else (oc - base)
            tests[metric][b] = report(d, f"vs {b}")
    payload["baseline_tests"] = tests

    print("\n" + "=" * 78)
    print("K ablation -- K=60 / K=80 against the chosen K=40, same folds")
    print("=" * 78)
    keys40, _ = per_fold(main_file, "OmicsCouncil", "accuracy")
    k_tests: Dict[str, Dict] = {}
    for K in (60, 80):
        path = os.path.join(RESULTS, K_FILES[K])
        print(f"\n--- K={K} ---")
        k_tests[f"K{K}"] = {}
        for metric in ["accuracy", "macro_f1", "ece"]:
            keys, vk = per_fold(path, "OmicsCouncil", metric)
            assert keys == keys40, "folds are not aligned across K runs"
            _, v40 = per_fold(main_file, "OmicsCouncil", metric)
            d = (v40 - vk) if metric in LOWER_IS_BETTER else (vk - v40)
            k_tests[f"K{K}"][metric] = report(d, f"K={K} better than K=40 ({metric})")
    payload["k_ablation"] = k_tests

    print("\n" + "=" * 78)
    print("Per-K summary (OmicsCouncil)")
    print("=" * 78)
    print(f"  {'K':>3}  {'accuracy':>14}  {'macro-F1':>9}  {'ECE':>7}  "
          f"{'faithfulness':>13}  {'containment':>11}")
    summary = {}
    for K, fname in K_FILES.items():
        blob = json.load(open(os.path.join(RESULTS, fname), encoding="utf-8"))
        oc = next(r for r in blob["summary"] if r["model"] == "OmicsCouncil")
        t = blob["trustworthy"]
        summary[f"K{K}"] = {
            "accuracy_mean": oc["accuracy_mean"], "accuracy_std": oc["accuracy_std"],
            "macro_f1_mean": oc["macro_f1_mean"], "ece_mean": oc["ece_mean"],
            "faithfulness_mean": t["faithfulness_mean"],
            "containment": t["halluc_containment_rate"],
        }
        print(f"  {K:>3}  {oc['accuracy_mean']:.3f}+/-{oc['accuracy_std']:.3f}  "
              f"{oc['macro_f1_mean']:>9.3f}  {oc['ece_mean']:>7.3f}  "
              f"{t['faithfulness_mean']:>13.3f}  "
              f"{t['halluc_containment_rate']:>11.3f}")
    payload["k_summary"] = summary

    out = os.path.join(RESULTS, "significance_20x5.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
