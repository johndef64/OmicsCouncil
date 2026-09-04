"""5-fold stratified CV variant of run_classification.py.

Reports mean ± std for every classification metric across the folds.
Trustworthy metrics (containment, faithfulness, marker recovery) are
aggregated across all test traces of all folds, so they reflect the
behaviour over the whole cohort.

    python experiments/run_classification_cv.py --config configs/tcga_brca.yaml --folds 5
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from _common import (DEFAULT_CONFIG, ConcatBaseline, ConcatMLPBaseline,
                     LateFusionBaseline, _ROOT, print_table, save_json)

from omicscouncil.config import load_config
from omicscouncil.data import MultiModalDataset, build_multimodal_dataset
from omicscouncil.metrics import (classification_metrics,
                                   expected_calibration_error, faithfulness,
                                   hallucination_containment, marker_recovery)
from omicscouncil.pipeline import OmicsCouncilPipeline


def _subset(ds: MultiModalDataset, idx: np.ndarray) -> MultiModalDataset:
    return ds.subset(idx)


def _eval_baseline(name: str, model, test: MultiModalDataset) -> Dict[str, float]:
    proba = model.predict_proba(test)
    m = classification_metrics(test.y, proba.argmax(axis=1))
    m["ece"] = expected_calibration_error(test.y, proba)
    m["model"] = name
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="classification_cv.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = build_multimodal_dataset(cfg)
    print(f"Domain: {cfg.domain} | N={ds.n_samples} | classes={ds.class_names}")
    print(f"Running {args.folds}-fold stratified CV (seed={cfg.seed})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg.seed)

    fold_records: List[Dict] = []                           # per-fold metrics
    all_oc_traces = []                                      # for global trustworthy
    faith_per_fold: List[float] = []                        # one value per fold

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(ds.n_samples), ds.y), start=1):
        print(f"\n--- Fold {fold_idx}/{args.folds}  (train={len(tr_idx)}, test={len(te_idx)}) ---")
        train = _subset(ds, tr_idx)
        test = _subset(ds, te_idx)

        # OmicsCouncil
        pipe = OmicsCouncilPipeline(cfg).fit(train)
        proba, traces = pipe.predict_proba(test)
        oc = classification_metrics(test.y, proba.argmax(axis=1))
        oc["ece"] = expected_calibration_error(test.y, proba)
        oc["model"] = "OmicsCouncil"
        oc["fold"] = fold_idx

        # Baselines on the same fold
        rows = [oc]
        for name, M in [
            ("Concat+LogReg", ConcatBaseline(max_iter=cfg.classifier.max_iter)),
            ("Concat+MLP",    ConcatMLPBaseline(max_iter=400)),
            ("Late-fusion",   LateFusionBaseline(max_iter=cfg.classifier.max_iter)),
        ]:
            M.fit(train)
            r = _eval_baseline(name, M, test)
            r["fold"] = fold_idx
            rows.append(r)
        fold_records.extend(rows)

        all_oc_traces.extend(traces)
        # Faithfulness is computed on every fold, not just the last one:
        # it needs the fold's own fitted head to re-encode its traces.
        faith_per_fold.append(float(faithfulness(pipe, proba, traces, top_k=3)))

    # ---- Aggregate per-model: mean +/- std across folds -----------------
    def _agg(model_name: str) -> Dict[str, float]:
        rows = [r for r in fold_records if r["model"] == model_name]
        out = {"model": model_name}
        for k in ("accuracy", "macro_f1", "balanced_accuracy", "ece"):
            vals = [r[k] for r in rows]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"]  = float(np.std(vals))
            out[f"{k}_str"]  = f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
        return out

    models = ["Concat+LogReg", "Concat+MLP", "Late-fusion", "OmicsCouncil"]
    summary = [_agg(m) for m in models]

    print_table(
        f"{args.folds}-fold CV — mean ± std",
        summary,
        cols=["model", "accuracy_str", "macro_f1_str", "balanced_accuracy_str", "ece_str"],
    )

    # ---- Trustworthy metrics across all OC traces -----------------------
    trust = {}
    trust.update({f"halluc_{k}": v for k, v in hallucination_containment(all_oc_traces).items()})
    trust.update({f"marker_{k}": v for k, v in marker_recovery(all_oc_traces, cfg.reference_markers).items()})
    if faith_per_fold:
        trust["faithfulness_mean"] = float(np.mean(faith_per_fold))
        trust["faithfulness_std"] = float(np.std(faith_per_fold, ddof=1)) \
            if len(faith_per_fold) > 1 else 0.0

    print("\nTrustworthy metrics (pooled across all folds):")
    for k, v in trust.items():
        print(f"  {k:<32}  {v:.4f}")

    payload = {
        "config": args.config,
        "folds": args.folds,
        "summary": summary,
        "trustworthy": trust,
        "per_fold": fold_records,
    }
    out = save_json(args.out, payload)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
