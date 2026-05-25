"""Ablation study: each protocol component is switched off in turn to isolate
its contribution. The switches are the same config flags the framework uses in
production, so an ablation is just a modified config.

    python experiments/run_ablation.py --config configs/breast_cancer.yaml
"""

from __future__ import annotations

import argparse
import copy

from _common import DEFAULT_CONFIG, load_split, print_table, save_json

from omicscouncil.metrics import (classification_metrics,
                                   expected_calibration_error,
                                   hallucination_containment)
from omicscouncil.pipeline import OmicsCouncilPipeline

VARIANTS = {
    "full":          dict(enable_cross_examination=True,  enable_arbiter=True,  typed_claims=True),
    "no_cross_exam": dict(enable_cross_examination=False, enable_arbiter=True,  typed_claims=True),
    "no_arbiter":    dict(enable_cross_examination=True,  enable_arbiter=False, typed_claims=True),
    "untyped":       dict(enable_cross_examination=True,  enable_arbiter=True,  typed_claims=False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default="ablation.json")
    args = ap.parse_args()

    cfg, train, test = load_split(args.config)
    rows, payload = [], {}

    for name, flags in VARIANTS.items():
        c = copy.deepcopy(cfg)
        for k, v in flags.items():
            setattr(c.council, k, v)

        pipe = OmicsCouncilPipeline(c).fit(train)
        proba, traces = pipe.predict_proba(test)
        m = classification_metrics(test.y, proba.argmax(axis=1))
        m["ece"] = expected_calibration_error(test.y, proba)
        m["containment_rate"] = hallucination_containment(traces)["containment_rate"]
        m["variant"] = name
        rows.append(m)
        payload[name] = m

    print_table("Ablation (test set)", rows,
                cols=["variant", "accuracy", "macro_f1", "ece", "containment_rate"])
    print(f"\nSaved -> {save_json(args.out, payload)}")


if __name__ == "__main__":
    main()
