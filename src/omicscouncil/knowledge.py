"""Knowledge prior and the graph-mediated arbiter (Round 3).

The arbiter grades each claim and the protocol drops the weakly-anchored
ones — this is the hallucination-containment mechanism.

Two sources are supported, with different arbitration semantics:

* ``derive_from_data`` (lightweight Wisconsin path) — a directed
  feature->class association graph estimated on the training split. The
  arbiter does exact ``(subject, relation, object)`` lookup against this
  prior, mapping weight terciles to grades E1..E3 plus E4 for explicit
  contradictions. By construction the prior agrees with the agents and the
  containment rate is near zero; we keep this path as a deliberately-
  documented limitation and a cross-scale ablation.

* ``pkt_subset`` (medium-weight TCGA path, "Option C") — a biomedical
  knowledge graph subset built from PheKnowLator
  (``data/kg/PKT/brca_subset.json``). Here the arbiter ignores the
  ``(relation, object)`` part of the claim — PKT does not encode
  feature->class facts — and grades the ``subject`` (HGNC gene symbol) by
  biological anchoring: groundedness, pathway / GO membership, disease
  association. Containment is non-zero by construction because a fraction
  of features do not map to any PKT gene node.
"""

from __future__ import annotations

import json
from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from .claims import Claim
from .config import KnowledgeConfig
from .data import MultiModalDataset

EdgeKey = Tuple[str, str, str]   # (subject, relation, object)


class KnowledgePrior:
    """Directed multigraph of typed triples plus an arbitration rule.

    Carries one of two mutually exclusive arbitration modes:

      * ``"weighted_lookup"``  — used by ``derive_from_data``.
      * ``"pkt_anchored"``     — used by ``from_pkt_subset``.
    """

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.edges: Dict[EdgeKey, float] = {}
        self._q_low = 0.0
        self._q_high = 1.0

        # Mode + per-symbol anchoring flags (populated by from_pkt_subset).
        self.mode: str = "weighted_lookup"
        self._grounded: Set[str] = set()
        self._has_pathway: Set[str] = set()
        self._has_go: Set[str] = set()
        self._has_disease: Set[str] = set()

    # ------------------------------------------------------------------ build
    def _add_edge(self, subject: str, relation: str, obj: str, weight: float) -> None:
        self.edges[(subject, relation, obj)] = weight
        self.graph.add_edge(subject, obj, key=relation, relation=relation, weight=weight)

    def _finalize_weights(self) -> None:
        """Cache weight terciles used to map weights to evidence grades."""
        if self.edges:
            w = np.array(list(self.edges.values()), dtype=float)
            self._q_low, self._q_high = np.quantile(w, [0.33, 0.66])

    @classmethod
    def derive_from_data(cls, train: MultiModalDataset) -> "KnowledgePrior":
        """Estimate feature<->class associations across all modalities.

        Lightweight, data-derived prior used by the Wisconsin instantiation.
        Arbiter mode: ``weighted_lookup``.
        """
        kp = cls()
        kp.mode = "weighted_lookup"
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
        if kp.edges:
            wmax = max(kp.edges.values()) + 1e-8
            for k in kp.edges:
                kp.edges[k] /= wmax
                s, r, o = k
                kp.graph[s][o][r]["weight"] = kp.edges[k]
        kp._finalize_weights()
        return kp

    @classmethod
    def from_file(cls, path: str) -> "KnowledgePrior":
        """Load an external KG edge list as a weighted-lookup prior.

        Expected JSON shape: ``[{subject, relation, object, weight?}, ...]``.
        Used for non-PKT external priors that share the data-derived
        ``(subject, relation, object)`` semantics.
        """
        kp = cls()
        kp.mode = "weighted_lookup"
        with open(path, "r", encoding="utf-8") as fh:
            for e in json.load(fh):
                kp._add_edge(e["subject"], e["relation"], e["object"],
                             float(e.get("weight", 1.0)))
        kp._finalize_weights()
        return kp

    @classmethod
    def from_pkt_subset(cls, path: str) -> "KnowledgePrior":
        """Load a PheKnowLator subset (``brca_subset.json``) and switch the
        arbiter to ``pkt_anchored`` mode.

        Records every triple in the underlying graph (used by ``degree``) but
        the arbiter consults only per-symbol anchoring flags pre-computed
        here, never doing edge lookup by ``(subject, relation, object)``.
        """
        kp = cls()
        kp.mode = "pkt_anchored"
        with open(path, "r", encoding="utf-8") as fh:
            triples = json.load(fh)

        for tr in triples:
            subj = tr["subject"]
            obj = tr["object"]
            rel = tr.get("relation", "related_to")
            obj_type = tr.get("object_type", "gene")
            kp._add_edge(subj, rel, obj, float(tr.get("weight", 1.0)))
            kp._grounded.add(subj)
            if obj_type == "gene":
                # Induced gene-gene edge — anchors both endpoints.
                kp._grounded.add(obj)
            if obj_type == "pathway":
                kp._has_pathway.add(subj)
            elif obj_type == "go":
                kp._has_go.add(subj)
            elif obj_type == "disease":
                kp._has_disease.add(subj)
        kp._finalize_weights()
        return kp

    # ------------------------------------------------------------- arbitrate
    def arbitrate(self, claim: Claim, typed: bool = True) -> str:
        """Assign an evidence grade in {E1, E2, E3, E4}.

        Dispatches on the prior's ``mode``:

        * ``weighted_lookup``  — exact ``(subject, relation, object)`` lookup;
          terciles of edge weights map to E1 (strong) / E2 (moderate) /
          E3 (unsupported) and an explicit-contradiction check assigns E4.
        * ``pkt_anchored``     — ignores ``(relation, object)``; grades the
          ``subject`` (HGNC symbol) by anchoring flags. E1 if grounded with
          pathway or GO membership, E2 if grounded without either, E3 if
          not grounded in PKT. E4 is reserved for explicitly-flagged
          non-coding / pseudogene subjects (not populated yet).
        """
        if not typed or claim.relation == "associated_with":
            # Untyped claims carry no verifiable subject typing.
            return "E2"

        if self.mode == "pkt_anchored":
            return self._arbitrate_pkt(claim)
        return self._arbitrate_lookup(claim)

    def _arbitrate_pkt(self, claim: Claim) -> str:
        subj = claim.subject
        if subj not in self._grounded:
            return "E3"
        if subj in self._has_pathway or subj in self._has_go:
            return "E1"
        return "E2"

    def _arbitrate_lookup(self, claim: Claim) -> str:
        w = self.edges.get((claim.subject, claim.relation, claim.object))
        if w is not None:
            if w >= self._q_high:
                return "E1"
            if w >= self._q_low:
                return "E2"
            return "E3"
        for (s, r, o), ww in self.edges.items():
            if s == claim.subject and r == claim.relation and o != claim.object:
                if ww >= self._q_high:
                    return "E4"
        return "E3"

    # --------------------------------------------------------- introspection
    def degree(self, node: str) -> int:
        return self.graph.degree(node) if node in self.graph else 0

    def is_grounded(self, symbol: str) -> bool:
        if self.mode == "pkt_anchored":
            return symbol in self._grounded
        return symbol in self.graph

    def grounding_stats(self) -> Dict[str, int]:
        """Diagnostics for the PKT-anchored mode."""
        return {
            "grounded": len(self._grounded),
            "with_pathway": len(self._has_pathway),
            "with_go": len(self._has_go),
            "with_disease": len(self._has_disease),
        }


def build_knowledge_prior(cfg: KnowledgeConfig, train: MultiModalDataset) -> KnowledgePrior:
    if cfg.source == "derive_from_data":
        return KnowledgePrior.derive_from_data(train)
    if cfg.source == "pkt_subset":
        if not cfg.path:
            raise ValueError("knowledge.source='pkt_subset' requires knowledge.path")
        return KnowledgePrior.from_pkt_subset(cfg.path)
    if cfg.source == "file":
        if not cfg.path:
            raise ValueError("knowledge.source='file' requires knowledge.path")
        return KnowledgePrior.from_file(cfg.path)
    raise ValueError(f"Unknown knowledge source '{cfg.source}'.")
