"""Encoding the deliberation trace into a fixed-length symbolic vector (Z-sym).

This is the representation the downstream classifier consumes. Crucially, every
dimension has an explicit meaning (claims of relation r toward class c, claims
from modality m, claims at evidence grade g, ...), so a prediction can always be
traced back to the verified claims that produced it — interpretability by
construction, with no post-hoc attribution.

The schema is derived entirely from the config (relations, classes, modalities),
so it adapts to a new domain automatically.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .claims import DeliberationTrace
from .knowledge import KnowledgePrior

# Evidence grades that contribute interpretable count features.
_GRADES = ["E1", "E2"]
_FREEFORM_RELATION = "associated_with"


class RepresentationSchema:
    """Deterministic mapping from a trace to a fixed index layout."""

    def __init__(self, relations: List[str], class_names: List[str], modalities: List[str]):
        # Always include the free-form relation so the untyped ablation fits the
        # same vector layout and stays comparable.
        self.relations = list(relations)
        if _FREEFORM_RELATION not in self.relations:
            self.relations = self.relations + [_FREEFORM_RELATION]
        self.class_names = list(class_names)
        self.modalities = list(modalities)

        self._labels: List[str] = []
        for r in self.relations:
            for c in self.class_names:
                self._labels.append(f"count[{r}->{c}]")
                self._labels.append(f"conf[{r}->{c}]")
        for m in self.modalities:
            self._labels.append(f"nclaims[{m}]")
        for g in _GRADES:
            self._labels.append(f"grade[{g}]")
        self._labels.append("total_verified")
        self._labels.append("mean_kg_degree")

    @property
    def dim(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def encode(self, trace: DeliberationTrace, prior: KnowledgePrior) -> np.ndarray:
        v = np.zeros(self.dim, dtype=float)
        idx = {lab: i for i, lab in enumerate(self._labels)}

        degrees = []
        for c in trace.verified_claims:
            ck = f"count[{c.relation}->{c.object}]"
            fk = f"conf[{c.relation}->{c.object}]"
            if ck in idx:
                v[idx[ck]] += 1.0
                v[idx[fk]] += c.confidence
            mk = f"nclaims[{c.modality}]"
            if mk in idx:
                v[idx[mk]] += 1.0
            if c.grade in _GRADES:
                v[idx[f"grade[{c.grade}]"]] += 1.0
            degrees.append(prior.degree(c.subject))

        v[idx["total_verified"]] = float(len(trace.verified_claims))
        v[idx["mean_kg_degree"]] = float(np.mean(degrees)) if degrees else 0.0
        return v


def build_schema(relations, class_names, modalities) -> RepresentationSchema:
    return RepresentationSchema(relations, class_names, modalities)
