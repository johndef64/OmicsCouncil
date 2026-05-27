"""Main experiment: train OmicsCouncil, evaluate subtype/diagnosis classification
and the trustworthy-representation metrics, and dump one example trace.

    python experiments/run_classification.py --config configs/breast_cancer.yaml
"""

from __future__ import annotations

import argparse

from _common import (DEFAULT_CONFIG, ConcatBaseline, ConcatMLPBaseline,
                     LateFusionBaseline, load_split, print_table, save_json)

from omicscouncil.metrics import (classification_metrics,
                                   expected_calibration_error, faithfulness,
                                   hallucination_containment, marker_recovery)
from omicscouncil.pipeline import OmicsCouncilPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="classification.json")
    args = ap.parse_args()

    cfg, train, test = load_split(args.config)
    print(f"Domain: {cfg.domain} | modalities: {cfg.modality_names} | "
          f"train={train.n_samples} test={test.n_samples}")

    # --- OmicsCouncil ------------------------------------------------------
    pipe = OmicsCouncilPipeline(cfg).fit(train)
    proba, traces = pipe.predict_proba(test)
    y_pred = proba.argmax(axis=1)

    oc = classification_metrics(test.y, y_pred)
    oc["ece"] = expected_calibration_error(test.y, proba)
    oc["faithfulness"] = faithfulness(pipe, proba, traces, top_k=3)
    oc.update({f"halluc_{k}": v for k, v in hallucination_containment(traces).items()})
    oc.update({f"marker_{k}": v for k, v in marker_recovery(traces, cfg.reference_markers).items()})

    # --- Baselines --------------------------------------------------------
    baselines = {
        "Concat+LogReg":  ConcatBaseline(max_iter=cfg.classifier.max_iter).fit(train),
        "Concat+MLP":     ConcatMLPBaseline(max_iter=400).fit(train),
        "Late-fusion":    LateFusionBaseline(max_iter=cfg.classifier.max_iter).fit(train),
    }
    baseline_rows = []
    baseline_metrics = {}
    for name, m in baselines.items():
        proba_b = m.predict_proba(test)
        bm = classification_metrics(test.y, proba_b.argmax(axis=1))
        bm["ece"] = expected_calibration_error(test.y, proba_b)
        baseline_rows.append({"model": name, **bm})
        baseline_metrics[name] = bm

    print_table(
        "Classification (test set)",
        baseline_rows + [{"model": "OmicsCouncil (Z-sym)", **oc}],
        cols=["model", "accuracy", "macro_f1", "balanced_accuracy", "ece"],
    )
    print_table(
        "Trustworthy-representation metrics (OmicsCouncil)",
        [{"metric": "faithfulness (conf drop, top-3 claims)", "value": oc["faithfulness"]},
         {"metric": "hallucination containment rate", "value": oc["halluc_containment_rate"]},
         {"metric": "pruned claims / sample", "value": oc["halluc_pruned_per_sample"]},
         {"metric": "marker recovery (>=1)", "value": oc["marker_recovery_ge1"]},
         {"metric": "marker recovery (>=3)", "value": oc["marker_recovery_ge3"]}],
        cols=["metric", "value"],
    )

    # --- One qualitative trace --------------------------------------------
    print("\n" + traces[0].as_report())

    out = save_json(args.out, {"config": args.config, "omicscouncil": oc,
                               "baselines": baseline_metrics,
                               "baseline": baseline_metrics.get("Concat+LogReg", {}),
                               "representation_dim": pipe.schema.dim})
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
