"""Evaluation metrics.

Beyond standard classification scores we report the metrics REHMED cares about
for trustworthy representations: calibration (ECE), missing-modality robustness
(driven from ``experiments/run_robustness.py``), and two interpretability
measures — marker recovery and faithfulness — plus a hallucination-containment
rate that quantifies how much the arbiter prunes.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from .claims import DeliberationTrace
from .pipeline import OmicsCouncilPipeline


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE over the predicted (max-probability) class."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def marker_recovery(traces: List[DeliberationTrace], reference_markers: List[str],
                    k: int = 3) -> Dict[str, float]:
    """Fraction of traces whose verified claims mention reference domain markers."""
    ref = set(m.lower() for m in reference_markers)
    if not ref:
        return {"recovery_ge1": 0.0, f"recovery_ge{k}": 0.0, "mean_markers": 0.0}
    ge1 = gek = 0
    totals = []
    for t in traces:
        subjects = {c.subject.lower() for c in t.verified_claims}
        hits = len(subjects & ref)
        totals.append(hits)
        ge1 += int(hits >= 1)
        gek += int(hits >= k)
    n = max(len(traces), 1)
    return {
        "recovery_ge1": ge1 / n,
        f"recovery_ge{k}": gek / n,
        "mean_markers": float(np.mean(totals)) if totals else 0.0,
    }


def hallucination_containment(traces: List[DeliberationTrace]) -> Dict[str, float]:
    """How aggressively the arbiter prunes contested claims.

    Containment rate = pruned / (pruned + arbiter-verified contested claims),
    i.e. the share of grounded-and-arbitrated evidence the prior rejects.
    """
    pruned = sum(len(t.pruned_claims) for t in traces)
    verified = sum(len(t.verified_claims) for t in traces)
    denom = pruned + verified
    return {
        "pruned_per_sample": pruned / max(len(traces), 1),
        "verified_per_sample": verified / max(len(traces), 1),
        "containment_rate": (pruned / denom) if denom else 0.0,
    }


def faithfulness(pipeline: OmicsCouncilPipeline, proba: np.ndarray,
                 traces: List[DeliberationTrace], top_k: int = 3) -> float:
    """Mean drop in predicted-class probability when the top-k verified claims
    are removed from the trace and the representation is recomputed.

    A larger drop means the prediction genuinely rests on those claims — the
    representation is faithful to the evidence it exposes.
    """
    assert pipeline.schema is not None and pipeline.prior is not None
    base_pred = proba.argmax(axis=1)
    base_conf = proba[np.arange(len(proba)), base_pred]

    ablated_vectors = []
    for t in traces:
        t2 = copy.deepcopy(t)
        t2.verified_claims = sorted(
            t2.verified_claims, key=lambda c: -c.confidence
        )[top_k:]
        ablated_vectors.append(pipeline.schema.encode(t2, pipeline.prior))

    ablated_proba = pipeline.clf.predict_proba(np.vstack(ablated_vectors))
    ablated_conf = ablated_proba[np.arange(len(ablated_proba)), base_pred]
    return float(np.mean(base_conf - ablated_conf))
