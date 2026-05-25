"""Knowledge prior and the graph-mediated arbiter (Round 3).

The arbiter is the component that grounds the deliberation in knowledge an
individual agent does not have: while each agent sees only its own modality, the
prior is built from *cross-modal* association (or, in deployment, from a curated
biomedical KG). It assigns every claim an evidence grade E1..E4 and the protocol
drops the weak ones — this is the hallucination-containment mechanism.

Two interchangeable sources:
  * ``derive_from_data`` — a feature<->class association graph estimated on the
    training split. A lightweight, dependency-free stand-in for a real KG.
  * ``file`` — an external edge list (JSON), e.g. a Reactome/STRING/PrimeKG
    export, with no change to the arbitration logic.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .claims import Claim
from .config import KnowledgeConfig
from .data import MultiModalDataset

EdgeKey = Tuple[str, str, str]   # (subject, relation, object)


class KnowledgePrior:
    """A directed multigraph of typed (subject, relation, object) edges with
    association weights, plus the arbitration rule over it."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.edges: Dict[EdgeKey, float] = {}
        self._q_low = 0.0
        self._q_high = 1.0

    # ------------------------------------------------------------------ build
    def _add_edge(self, subject: str, relation: str, obj: str, weight: float) -> None:
        self.edges[(subject, relation, obj)] = weight
        self.graph.add_edge(subject, obj, key=relation, relation=relation, weight=weight)

    def _finalize(self) -> None:
        """Cache weight terciles used to map weights to evidence grades."""
        if self.edges:
            w = np.array(list(self.edges.values()), dtype=float)
            self._q_low, self._q_high = np.quantile(w, [0.33, 0.66])

    @classmethod
    def derive_from_data(cls, train: MultiModalDataset) -> "KnowledgePrior":
        """Estimate feature<->class directional associations across all modalities."""
        kp = cls()
        y = train.y
        for mod in train.modalities.values():
            X = mod.X
            mu = X.mean(axis=0)
            sigma = X.std(axis=0) + 1e-8
            Z = (X - mu) / sigma
            for j, fname in enumerate(mod.feature_names):
                class_means = {c: Z[y == ci, j].mean()
                               for ci, c in enumerate(train.class_names)}
                hi = max(class_means, key=class_means.get)
                lo = min(class_means, key=class_means.get)
                sep = abs(class_means[hi] - class_means[lo])
                kp._add_edge(fname, "elevated_in", hi, sep)
                kp._add_edge(fname, "reduced_in", lo, sep)
        # Normalize weights to [0, 1] for stable, dataset-independent thresholds.
        if kp.edges:
            wmax = max(kp.edges.values()) + 1e-8
            for k in kp.edges:
                kp.edges[k] /= wmax
                s, r, o = k
                kp.graph[s][o][r]["weight"] = kp.edges[k]  # keep graph in sync
        kp._finalize()
        return kp

    @classmethod
    def from_file(cls, path: str) -> "KnowledgePrior":
        """Load an external KG edge list: [{subject, relation, object, weight}, ...]."""
        kp = cls()
        with open(path, "r", encoding="utf-8") as fh:
            for e in json.load(fh):
                kp._add_edge(e["subject"], e["relation"], e["object"],
                             float(e.get("weight", 1.0)))
        kp._finalize()
        return kp

    # ------------------------------------------------------------- arbitrate
    def arbitrate(self, claim: Claim, typed: bool = True) -> str:
        """Assign an evidence grade in {E1, E2, E3, E4}.

        E1 strong support · E2 moderate/uncertain · E3 unsupported ·
        E4 contradicted by the prior.
        """
        if not typed or claim.relation == "associated_with":
            # Untyped claims cannot be looked up; the arbiter cannot discriminate.
            return "E2"

        w = self.edges.get((claim.subject, claim.relation, claim.object))
        if w is not None:
            if w >= self._q_high:
                return "E1"
            if w >= self._q_low:
                return "E2"
            return "E3"

        # No matching edge: is there a strong edge that contradicts this claim?
        for (s, r, o), ww in self.edges.items():
            if s == claim.subject and r == claim.relation and o != claim.object:
                if ww >= self._q_high:
                    return "E4"
        return "E3"

    def degree(self, node: str) -> int:
        return self.graph.degree(node) if node in self.graph else 0


def build_knowledge_prior(cfg: KnowledgeConfig, train: MultiModalDataset) -> KnowledgePrior:
    if cfg.source == "derive_from_data":
        return KnowledgePrior.derive_from_data(train)
    if cfg.source == "file":
        if not cfg.path:
            raise ValueError("knowledge.source='file' requires knowledge.path")
        return KnowledgePrior.from_file(cfg.path)
    raise ValueError(f"Unknown knowledge source '{cfg.source}'.")
