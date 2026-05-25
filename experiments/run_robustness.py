"""Missing-modality robustness: drop 0, 1, 2 modalities and measure how macro-F1
degrades. OmicsCouncil simply omits the silent agent; the early-integration
baseline must mean-impute the missing block. We average over all modality
subsets of each size.

    python experiments/run_robustness.py --config configs/breast_cancer.yaml
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import List

import numpy as np

from _common import (DEFAULT_CONFIG, ConcatBaseline, load_split, print_table,
                     save_json)

from omicscouncil.metrics import classification_metrics
from omicscouncil.pipeline import OmicsCouncilPipeline


def _avg_f1_over_drops(predict_fn, y_true, all_mods: List[str], n_drop: int) -> float:
    """Average macro-F1 over every way of dropping ``n_drop`` modalities."""
    kept = len(all_mods) - n_drop
    f1s = []
    for active in combinations(all_mods, kept):
        y_pred = predict_fn(list(active))
        f1s.append(classification_metrics(y_true, y_pred)["macro_f1"])
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="robustness.json")
    args = ap.parse_args()

    cfg, train, test = load_split(args.config)
    mods = train.modality_names

    pipe = OmicsCouncilPipeline(cfg).fit(train)
    base = ConcatBaseline(max_iter=cfg.classifier.max_iter).fit(train)

    def oc_predict(active):
        return pipe.predict(test, active_modalities=active)[0]

    def base_predict(active):
        return base.predict(test, active=active)

    rows, payload = [], {}
    for n_drop in range(0, len(mods)):
        oc_f1 = _avg_f1_over_drops(oc_predict, test.y, mods, n_drop)
        bs_f1 = _avg_f1_over_drops(base_predict, test.y, mods, n_drop)
        row = {"modalities_dropped": n_drop,
               "omicscouncil_macroF1": oc_f1,
               "baseline_macroF1": bs_f1}
        rows.append(row)
        payload[f"drop_{n_drop}"] = row

    # Retention ratio relative to the full-modality score.
    oc0 = rows[0]["omicscouncil_macroF1"]
    bs0 = rows[0]["baseline_macroF1"]
    for r in rows:
        r["oc_retention"] = r["omicscouncil_macroF1"] / oc0 if oc0 else 0.0
        r["base_retention"] = r["baseline_macroF1"] / bs0 if bs0 else 0.0

    print_table("Missing-modality robustness (avg over subsets)", rows,
                cols=["modalities_dropped", "omicscouncil_macroF1", "baseline_macroF1",
                      "oc_retention", "base_retention"])
    print(f"\nSaved -> {save_json(args.out, payload)}")


if __name__ == "__main__":
    main()
