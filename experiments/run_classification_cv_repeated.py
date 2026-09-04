"""Repeated stratified K-fold CV + paired significance tests.

This is the protocol the paper reports: 20 repetitions of stratified 5-fold CV
(100 fits), with paired tests on the differences between models. Results are
written to ``results/``.

Why *repeated* 5-fold rather than plain 10-fold
-----------------------------------------------
Raising K does not straightforwardly buy statistical power: k-fold estimates
are correlated because the training sets overlap, and the overlap *grows* with
K.  A naive paired t-test over more folds is therefore anti-conservative
(Dietterich 1998; Bengio & Grandvalet 2004: there is no unbiased estimator of
the variance of k-fold CV).  Two further reasons to keep K=5 here:

  * N=272 with Her2=34.  At K=10 a test fold holds ~27 samples, i.e. ~3.4 Her2,
    and per-fold macro-F1 becomes dominated by noise.  At K=5 a test fold holds
    ~54 samples (~6.8 Her2).
  * K=5 keeps the headline numbers directly comparable with the submitted paper.

So we repeat the 5-fold protocol R times with different shuffles and report:

  * repeat-level paired tests (n=R quasi-independent observations) -- primary;
  * the Nadeau-Bengio corrected resampled t-test over all R*K fold-level
    differences, which inflates the variance to account for train/test overlap;
  * a percentile bootstrap CI on the mean difference;
  * win rates at fold and repeat level.

Bonus: the MLP init seed is varied per repeat, so MLP initialisation variance
is characterised rather than confounded with fold variance (a caveat the
submitted paper declares in Sec. 4).  Faithfulness is computed on *every* fold
instead of the last one only.

Usage
-----
    python experiments/run_classification_cv_repeated.py \
        --config configs/tcga_brca.yaml --folds 5 --repeats 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

# --- import bootstrap ------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from _common import (ConcatBaseline, ConcatMLPBaseline,  # noqa: E402
                     LateFusionBaseline, print_table)
from omicscouncil.config import load_config                       # noqa: E402
from omicscouncil.data import (MultiModalDataset,                 # noqa: E402
                               build_multimodal_dataset)
from omicscouncil.metrics import (classification_metrics,         # noqa: E402
                                  expected_calibration_error, faithfulness,
                                  hallucination_containment, marker_recovery)
from omicscouncil.pipeline import OmicsCouncilPipeline            # noqa: E402

OUT_DIR = os.path.join(_ROOT, "results")
MODELS = ["Concat+LogReg", "Concat+MLP", "Late-fusion", "OmicsCouncil"]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "ece"]
LOWER_IS_BETTER = {"ece"}


# ---------------------------------------------------------------- statistics
def _paired_stats(better: List[float], oc: List[float], lower_better: bool,
                  n_folds: int | None = None) -> Dict[str, float]:
    """Paired comparison. `diff` is always oriented so that >0 means OC wins."""
    from scipy import stats

    d = np.array([(b - o) if lower_better else (o - b)
                  for b, o in zip(better, oc)], dtype=float)
    n = len(d)
    out: Dict[str, float] = {
        "n": n,
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)) if n > 1 else 0.0,
        "wins": int((d > 0).sum()),
        "win_rate": float((d > 0).mean()),
    }
    if n > 1 and d.std(ddof=1) > 0:
        t = stats.ttest_1samp(d, 0.0)
        out["t_stat"] = float(t.statistic)
        out["p_ttest"] = float(t.pvalue)
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            out["p_wilcoxon"] = float("nan")

        # Nadeau-Bengio corrected resampled t-test (train/test overlap).
        if n_folds:
            corr = 1.0 / n + 1.0 / (n_folds - 1)
            t_nb = d.mean() / np.sqrt(corr * d.var(ddof=1))
            out["t_nadeau_bengio"] = float(t_nb)
            out["p_nadeau_bengio"] = float(
                2 * stats.t.sf(abs(t_nb), df=n - 1))

        # Percentile bootstrap CI on the mean difference.
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(d, size=n, replace=True).mean()
                          for _ in range(10000)])
        out["boot_ci_lo"] = float(np.percentile(boots, 2.5))
        out["boot_ci_hi"] = float(np.percentile(boots, 97.5))
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "tcga_brca.yaml"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--out", default="cv_repeated.json")
    ap.add_argument("--fix-mlp-seed", action="store_true",
                    help="keep MLP random_state=42 (paper behaviour) instead of "
                         "varying it per repeat")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()
    ds = build_multimodal_dataset(cfg)
    print(f"Domain: {cfg.domain} | N={ds.n_samples} | classes={ds.class_names}")
    print(f"Dataset built in {time.time() - t0:.1f}s")
    print(f"Repeated CV: {args.repeats} repeats x {args.folds} folds "
          f"= {args.repeats * args.folds} fits\n")

    records: List[Dict] = []          # one row per (repeat, fold, model)
    all_oc_traces = []
    faith_per_fold: List[float] = []

    t_start = time.time()
    for rep in range(args.repeats):
        seed = args.base_seed + rep
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        mlp_seed = 42 if args.fix_mlp_seed else seed

        for fold_idx, (tr_idx, te_idx) in enumerate(
                skf.split(np.zeros(ds.n_samples), ds.y), start=1):
            train: MultiModalDataset = ds.subset(tr_idx)
            test: MultiModalDataset = ds.subset(te_idx)

            pipe = OmicsCouncilPipeline(cfg).fit(train)
            proba, traces = pipe.predict_proba(test)
            oc = classification_metrics(test.y, proba.argmax(axis=1))
            oc["ece"] = expected_calibration_error(test.y, proba)
            oc.update(model="OmicsCouncil", fold=fold_idx, repeat=rep, seed=seed)
            rows = [oc]

            all_oc_traces.extend(traces)
            faith_per_fold.append(float(faithfulness(pipe, proba, traces, top_k=3)))

            for name, M in [
                ("Concat+LogReg", ConcatBaseline(max_iter=cfg.classifier.max_iter)),
                ("Concat+MLP",    ConcatMLPBaseline(max_iter=600, alpha=1e-2,
                                                    random_state=mlp_seed)),
                ("Late-fusion",   LateFusionBaseline(max_iter=cfg.classifier.max_iter)),
            ]:
                M.fit(train)
                p = M.predict_proba(test)
                r = classification_metrics(test.y, p.argmax(axis=1))
                r["ece"] = expected_calibration_error(test.y, p)
                r.update(model=name, fold=fold_idx, repeat=rep, seed=seed)
                rows.append(r)

            records.extend(rows)

        done = (rep + 1) * args.folds
        el = time.time() - t_start
        print(f"  repeat {rep + 1}/{args.repeats} done "
              f"({done} fits, {el:.0f}s elapsed, {el / done:.1f}s/fit)")

        # Checkpoint after every repeat: these runs are long and a killed
        # process would otherwise lose everything. Raw records are enough to
        # rebuild the summary and the paired tests offline.
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, args.out + ".partial"), "w",
                  encoding="utf-8") as fh:
            json.dump({"repeats_done": rep + 1,
                       "config": args.config,
                       "folds": args.folds,
                       "per_fold": records,
                       "faithfulness_per_fold": faith_per_fold}, fh)

    # ---- aggregation -----------------------------------------------------
    def vals(model: str, metric: str) -> List[float]:
        return [r[metric] for r in records if r["model"] == model]

    def repeat_means(model: str, metric: str) -> List[float]:
        out = []
        for rep in range(args.repeats):
            v = [r[metric] for r in records
                 if r["model"] == model and r["repeat"] == rep]
            out.append(float(np.mean(v)))
        return out

    summary = []
    for m in MODELS:
        row = {"model": m}
        for k in METRICS:
            v = np.array(vals(m, k))
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_std"] = float(v.std(ddof=1))
            row[f"{k}_str"] = f"{v.mean():.3f}±{v.std(ddof=1):.3f}"
        summary.append(row)

    print_table(f"{args.repeats}x{args.folds} repeated CV - mean +/- std over "
                f"{args.repeats * args.folds} folds", summary,
                cols=["model", "accuracy_str", "macro_f1_str",
                      "balanced_accuracy_str", "ece_str"])

    # ---- paired tests, OmicsCouncil vs each baseline ---------------------
    tests: Dict[str, Dict[str, Dict]] = {}
    for metric in METRICS:
        lb = metric in LOWER_IS_BETTER
        tests[metric] = {}
        for base in MODELS[:3]:
            tests[metric][base] = {
                "fold_level": _paired_stats(vals(base, metric), vals(metric=metric, model="OmicsCouncil"),
                                            lb, n_folds=args.folds),
                "repeat_level": _paired_stats(repeat_means(base, metric),
                                              repeat_means("OmicsCouncil", metric), lb),
            }

    print("\nPaired tests (positive mean_diff = OmicsCouncil better)")
    for metric in METRICS:
        print(f"\n  === {metric} ===")
        for base in MODELS[:3]:
            f = tests[metric][base]["fold_level"]
            r = tests[metric][base]["repeat_level"]
            print(f"   vs {base:<14} folds {f['wins']:>2}/{f['n']:<3} "
                  f"d={f['mean_diff']:+.4f}  "
                  f"p_fold={f.get('p_ttest', float('nan')):.4f}  "
                  f"p_NB={f.get('p_nadeau_bengio', float('nan')):.4f}  "
                  f"p_repeat={r.get('p_ttest', float('nan')):.4f}  "
                  f"CI95=[{r.get('boot_ci_lo', float('nan')):+.4f},"
                  f"{r.get('boot_ci_hi', float('nan')):+.4f}]")

    # ---- trustworthy metrics --------------------------------------------
    trust: Dict[str, float] = {}
    trust.update({f"halluc_{k}": v for k, v in
                  hallucination_containment(all_oc_traces).items()})
    trust.update({f"marker_{k}": v for k, v in
                  marker_recovery(all_oc_traces, cfg.reference_markers).items()})
    trust["faithfulness_mean"] = float(np.mean(faith_per_fold))
    trust["faithfulness_std"] = float(np.std(faith_per_fold, ddof=1))
    trust["faithfulness_n_folds"] = len(faith_per_fold)

    print("\nTrustworthy metrics (pooled over all folds):")
    for k, v in trust.items():
        print(f"  {k:<32}  {v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "config": args.config,
        "protocol": f"{args.repeats}x{args.folds} repeated stratified CV",
        "folds": args.folds,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "mlp_seed_varied": not args.fix_mlp_seed,
        "n_fits": args.repeats * args.folds,
        "wall_seconds": time.time() - t_start,
        "summary": summary,
        "paired_tests": tests,
        "trustworthy": trust,
        "faithfulness_per_fold": faith_per_fold,
        "per_fold": records,
    }
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
