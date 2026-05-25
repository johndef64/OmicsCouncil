"""Modality-specific expert agents.

Each agent sees *only* its own modality. It plays two roles in the protocol:
  * Round 1 — :meth:`Agent.emit_claims` proposes typed evidence claims.
  * Round 2 — :meth:`Agent.react` cross-examines a peer's claim using its own
    modality's evidence.

The default :class:`StatisticalAgent` is fully deterministic, trains in
milliseconds, and needs no GPU or network — this is what keeps the whole
pipeline runnable in minutes on a laptop/CPU server. :class:`LLMAgent` is a
documented extension point: swapping it in turns the same protocol into the
full LLM-driven instantiation without touching the council logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from .claims import Claim, Reaction, ReactionType
from .config import AgentConfig
from .data import Modality


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = list(scores)
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = vals - vals.max()
    ex = np.exp(vals)
    ex = ex / ex.sum()
    return {k: float(v) for k, v in zip(keys, ex)}


class Agent(ABC):
    """Interface every modality expert must implement."""

    def __init__(self, modality_name: str, cfg: AgentConfig, typed: bool = True):
        self.modality = modality_name
        self.cfg = cfg
        self.typed = typed

    @abstractmethod
    def fit(self, modality: Modality, y: np.ndarray, class_names: List[str]) -> "Agent":
        ...

    @abstractmethod
    def emit_claims(self, x_row: np.ndarray) -> List[Claim]:
        ...

    @abstractmethod
    def class_scores(self, x_row: np.ndarray) -> Dict[str, float]:
        ...

    def react(self, claim: Claim, x_row: np.ndarray) -> Reaction:
        """Default cross-examination logic, shared by all agent types.

        The agent compares the claim's hypothesised class against the class
        distribution implied by *its own* modality.
        """
        scores = self.class_scores(x_row)
        probs = _softmax(scores)
        if not probs:
            return Reaction(claim.key(), self.modality, ReactionType.ABSTAIN, 0.0)

        top_class = max(probs, key=probs.get)
        p_obj = probs.get(claim.object, 0.0)
        p_top = probs[top_class]

        if claim.object == top_class and p_top >= 0.60:
            return Reaction(claim.key(), self.modality, ReactionType.SUPPORT, p_top)
        if claim.object == top_class:
            return Reaction(claim.key(), self.modality, ReactionType.EXTEND, p_top)
        if (p_top - p_obj) >= 0.40:
            return Reaction(claim.key(), self.modality, ReactionType.CONTRADICT, p_top)
        return Reaction(claim.key(), self.modality, ReactionType.ABSTAIN, p_obj)


class StatisticalAgent(Agent):
    """A transparent per-feature evidence reasoner.

    On ``fit`` it standardizes its modality and, for every feature, records the
    class with the highest and lowest class-conditional mean and the separation
    between them. At inference, for the most discriminative features, it reads
    the sample's standardized value and asserts a typed claim toward the class
    that an elevated (or reduced) value supports, with a calibrated confidence.
    """

    def fit(self, modality: Modality, y: np.ndarray, class_names: List[str]) -> "StatisticalAgent":
        self.feature_names = modality.feature_names
        self.class_names = class_names
        X = modality.X

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Z = (X - self.mu) / self.sigma

        n_features = X.shape[1]
        self.high_class = [""] * n_features
        self.low_class = [""] * n_features
        self.separation = np.zeros(n_features)

        for j in range(n_features):
            class_means = {c: Z[y == ci, j].mean() for ci, c in enumerate(class_names)}
            hi = max(class_means, key=class_means.get)
            lo = min(class_means, key=class_means.get)
            self.high_class[j] = hi
            self.low_class[j] = lo
            self.separation[j] = abs(class_means[hi] - class_means[lo])

        # Normalize separation to [0, 1] as a per-feature reliability weight.
        smax = self.separation.max() + 1e-8
        self.sep_norm = self.separation / smax

        # Keep only the most discriminative features for this modality.
        k = min(self.cfg.top_features_per_agent, n_features)
        self.selected = list(np.argsort(-self.separation)[:k])
        return self

    def _standardize(self, x_row: np.ndarray) -> np.ndarray:
        return (x_row - self.mu) / self.sigma

    def emit_claims(self, x_row: np.ndarray) -> List[Claim]:
        z = self._standardize(x_row)
        claims: List[Claim] = []
        for j in self.selected:
            margin = abs(z[j])
            conf = _sigmoid(self.cfg.confidence_beta * margin * (0.5 + self.sep_norm[j]))
            if conf < self.cfg.confidence_threshold:
                continue
            if z[j] >= 0:
                relation, obj = "elevated_in", self.high_class[j]
            else:
                relation, obj = "reduced_in", self.low_class[j]
            if not self.typed:
                relation = "associated_with"   # free-form ablation: loses groundability
            claims.append(Claim(
                subject=self.feature_names[j],
                relation=relation,
                object=obj,
                confidence=conf,
                modality=self.modality,
                evidence=float(z[j]),
            ))
        return claims

    def class_scores(self, x_row: np.ndarray) -> Dict[str, float]:
        z = self._standardize(x_row)
        scores = {c: 0.0 for c in self.class_names}
        for j in self.selected:
            margin = abs(z[j])
            conf = _sigmoid(self.cfg.confidence_beta * margin * (0.5 + self.sep_norm[j]))
            obj = self.high_class[j] if z[j] >= 0 else self.low_class[j]
            scores[obj] += conf
        return scores


class LLMAgent(Agent):
    """Optional LLM-backed agent (extension point, not required to run).

    To activate the full LLM instantiation described in the paper, implement the
    three methods below against an LLM backend (local Qwen/Llama or an API) that
    is prompted to read the modality and emit typed claims in the same
    :class:`~omicscouncil.claims.Claim` schema. The council and arbiter are
    backend-agnostic and require no changes.
    """

    def fit(self, modality: Modality, y: np.ndarray, class_names: List[str]) -> "LLMAgent":
        self.class_names = class_names
        raise NotImplementedError(
            "LLMAgent is a documented extension point. Provide an LLM backend "
            "that emits Claim objects to enable it; the default StatisticalAgent "
            "runs the full pipeline on CPU."
        )

    def emit_claims(self, x_row: np.ndarray) -> List[Claim]:  # pragma: no cover
        raise NotImplementedError

    def class_scores(self, x_row: np.ndarray) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError


def build_agent(modality_name: str, cfg: AgentConfig, typed: bool) -> Agent:
    """Factory selecting the agent backend from config."""
    if cfg.type == "statistical":
        return StatisticalAgent(modality_name, cfg, typed=typed)
    if cfg.type == "llm":
        return LLMAgent(modality_name, cfg, typed=typed)
    raise ValueError(f"Unknown agent type '{cfg.type}'.")
