"""Modality-specific expert agents.

Each agent sees *only* its own modality. It plays two roles in the protocol:
  * Round 1 — :meth:`Agent.emit_claims` proposes typed evidence claims.
  * Round 2 — :meth:`Agent.react` cross-examines a peer's claim using its own
    modality's evidence.

Two concrete agents:

* :class:`StatisticalAgent` — deterministic, transparent, CPU-only. Built on
  per-feature class-mean separations standardized within the modality.
* :class:`LLMAgent` — Groq-hosted Qwen3 produces typed claims from a textual
  per-feature modality profile + the sample's values. Disk-cached, with
  JSON-schema parsing and graceful fallback to *abstain* on parse failure.
  Swapping :class:`StatisticalAgent` for :class:`LLMAgent` does not require
  any change to :mod:`omicscouncil.council`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

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


# --------------------------------------------------------------------------- #
# Statistical agent (default, deterministic, CPU)
# --------------------------------------------------------------------------- #
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

        smax = self.separation.max() + 1e-8
        self.sep_norm = self.separation / smax

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
                relation = "associated_with"
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


# --------------------------------------------------------------------------- #
# LLM agent (Groq + Qwen)
# --------------------------------------------------------------------------- #
_LLM_CLIENT_READY = False


def _ensure_llm_client() -> None:
    """Make ``llm_lite_async`` importable and ensure the Groq env var is set.

    ``src/llm_lite_async.py`` lives at the same level as the ``omicscouncil``
    package; it reads ``api_keys.json`` from the cwd, which is not where we
    keep it. We populate the env var explicitly from ``src/api_keys.json``.
    """
    global _LLM_CLIENT_READY
    if _LLM_CLIENT_READY:
        return
    here = Path(__file__).resolve().parent.parent       # .../src
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    keys_path = here / "api_keys.json"
    if keys_path.exists() and not os.environ.get("GROQ_API_KEY"):
        with open(keys_path, "r", encoding="utf-8") as fh:
            keys = json.load(fh)
        if keys.get("groq"):
            os.environ["GROQ_API_KEY"] = keys["groq"]
    _LLM_CLIENT_READY = True


# A compact, modality-agnostic system prompt. We pin the claim grammar and
# show one few-shot example. The per-modality profile is sent in the user
# message so we can vary it per agent.
_LLM_SYSTEM_PROMPT = """/no_think
You are a modality-specific expert agent in a multi-agent council that classifies TCGA-BRCA samples into PAM50 subtypes. You see only ONE omic modality. Given a per-feature profile (which classes a feature is typically elevated / reduced in, based on training data) and the current sample's values, you emit typed evidence CLAIMS toward candidate PAM50 classes.

OUTPUT FORMAT (strict): a JSON array of objects with EXACTLY these keys:
  subject     (string, gene HGNC symbol from the input list)
  relation    ("elevated_in" or "reduced_in")
  object      (one of the candidate class names — case-sensitive)
  confidence  (float in (0, 1))

Rules:
- Use ONLY features that appear in the per-feature profile.
- relation MUST be "elevated_in" or "reduced_in" — nothing else.
- object MUST be one of the candidate classes.
- **Each feature must support a SINGLE class — the one whose mean is MOST distant from the others in the same direction as the sample's value**. Do NOT emit the same feature toward two different classes. Pick the most discriminative class only.
- Emit at most 25 claims; only the strongest evidence.
- Do not wrap the JSON in markdown fences. Do not add commentary.
- If you cannot identify any evidence above noise, return [].

Example output:
[{"subject":"ESR1","relation":"elevated_in","object":"LumA","confidence":0.86},
 {"subject":"KRT5","relation":"elevated_in","object":"Basal","confidence":0.78}]"""


class LLMAgent(Agent):
    """LLM-backed agent. Drop-in replacement for :class:`StatisticalAgent`.

    Fit-time: profiles each feature (class-conditional means + separation),
    keeps the top-K most discriminative, and caches a textual modality
    profile used as part of the prompt.

    Inference-time: builds a per-sample prompt (profile + sample's
    standardized values on the top-K features), calls Qwen3-32B via Groq,
    parses the JSON response into :class:`Claim`, and caches the raw
    response on disk so re-runs are free.

    Cross-examination (Round 2) reuses the statistical class_scores routine
    — the LLM is asked only to produce claims, the peer-react logic is
    framework-imposed and deterministic.
    """

    DEFAULT_MODEL = "qwen/qwen3-32b"
    DEFAULT_TEMP = 0.0
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_RETRIES = 1

    def __init__(self, modality_name: str, cfg: AgentConfig, typed: bool = True):
        super().__init__(modality_name, cfg, typed=typed)
        _ensure_llm_client()
        # Allow overrides via env (handy for CI / quick swaps).
        self.model = os.environ.get("OMICSCOUNCIL_LLM_MODEL", self.DEFAULT_MODEL)
        self.temperature = float(os.environ.get("OMICSCOUNCIL_LLM_TEMP", self.DEFAULT_TEMP))
        self.max_tokens = int(os.environ.get("OMICSCOUNCIL_LLM_MAX_TOKENS", self.DEFAULT_MAX_TOKENS))
        # Disk cache: keyed by hash(model, prompt). Lives in
        # `results/.llm_cache/` so it survives across runs.
        repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = repo_root / "results" / ".llm_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ fit
    def fit(self, modality: Modality, y: np.ndarray, class_names: List[str]) -> "LLMAgent":
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
        self.class_means_z = np.zeros((n_features, len(class_names)))

        for j in range(n_features):
            for ci, c in enumerate(class_names):
                self.class_means_z[j, ci] = Z[y == ci, j].mean()
            row = self.class_means_z[j]
            hi = int(np.argmax(row))
            lo = int(np.argmin(row))
            self.high_class[j] = class_names[hi]
            self.low_class[j] = class_names[lo]
            self.separation[j] = abs(row[hi] - row[lo])

        smax = self.separation.max() + 1e-8
        self.sep_norm = self.separation / smax

        k = min(self.cfg.top_features_per_agent, n_features)
        self.selected = list(np.argsort(-self.separation)[:k])

        # Build a once-and-done textual profile used as part of every prompt.
        self._profile_text = self._build_profile_text()
        return self

    def _build_profile_text(self) -> str:
        """Compact per-feature profile shown to the LLM.

        Format: ``SYMBOL: high={class} (mu=+1.2) low={class} (mu=-0.8)`` —
        one feature per line, only the top-K selected features. The LLM
        therefore sees the same discriminative information the statistical
        agent uses, but in natural language form.
        """
        lines = [f"Modality: {self.modality}",
                 f"Candidate classes: {', '.join(self.class_names)}",
                 "Top discriminative features (class-conditional standardized means):"]
        for j in self.selected:
            mus = self.class_means_z[j]
            parts = ", ".join(f"{c}={mu:+.2f}" for c, mu in zip(self.class_names, mus))
            lines.append(f"  {self.feature_names[j]}: {parts}")
        return "\n".join(lines)

    # ---------------------------------------------------------- prompt + io
    def _user_prompt(self, x_row: np.ndarray) -> str:
        z = (x_row - self.mu) / self.sigma
        lines = [self._profile_text, "",
                 "Current sample's standardized values on the top features:"]
        for j in self.selected:
            lines.append(f"  {self.feature_names[j]}: {z[j]:+.2f}")
        lines.append("")
        lines.append("Emit a JSON array of claims following the schema in the system prompt.")
        return "\n".join(lines)

    def _cache_key(self, user_prompt: str) -> Path:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(b"|")
        h.update(_LLM_SYSTEM_PROMPT.encode())
        h.update(b"|")
        h.update(user_prompt.encode())
        return self._cache_dir / f"{h.hexdigest()[:20]}.json"

    def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Cached, retried call to Groq. Returns the raw text or ``None``."""
        cache_path = self._cache_key(user_prompt)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        try:
            from llm_lite_async import llm
        except Exception as e:
            raise RuntimeError(f"llm_lite_async not importable: {e}")
        last_err = None
        for attempt in range(self.DEFAULT_RETRIES + 1):
            try:
                resp = llm(
                    user_prompt,
                    system=_LLM_SYSTEM_PROMPT,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                cache_path.write_text(resp, encoding="utf-8")
                return resp
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        # Cache the failure marker so we don't hammer the API on rerun.
        cache_path.write_text("__LLM_CALL_FAILED__", encoding="utf-8")
        sys.stderr.write(f"[LLMAgent] call failed for modality '{self.modality}': {last_err}\n")
        return None

    # ------------------------------------------------------------- parsing
    _JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

    def _parse_claims(self, raw: str, x_row: np.ndarray) -> List[Claim]:
        if not raw or raw == "__LLM_CALL_FAILED__":
            return []
        # Strip <think>...</think> blocks (Qwen3 may emit them).
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown fences if any.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        # Try the largest JSON array substring.
        m = self._JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            payload = json.loads(m.group(0))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        # Validate and convert.
        z = (x_row - self.mu) / self.sigma
        sym_to_idx = {s: i for i, s in enumerate(self.feature_names)}
        claims: List[Claim] = []
        valid_relations = {"elevated_in", "reduced_in"} if self.typed else {"associated_with"}
        valid_classes = set(self.class_names)

        for item in payload[:25]:                                # hard cap
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            rel = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            try:
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if subj not in sym_to_idx:
                continue
            if rel not in valid_relations:
                continue
            if obj not in valid_classes:
                continue
            conf = max(0.01, min(0.99, conf))
            if conf < self.cfg.confidence_threshold:
                continue
            if not self.typed:
                rel = "associated_with"
            j = sym_to_idx[subj]
            claims.append(Claim(
                subject=subj,
                relation=rel,
                object=obj,
                confidence=conf,
                modality=self.modality,
                evidence=float(z[j]),
            ))
        return claims

    # ------------------------------------------------------------- emit
    def emit_claims(self, x_row: np.ndarray) -> List[Claim]:
        prompt = self._user_prompt(x_row)
        raw = self._call_llm(prompt)
        return self._parse_claims(raw or "", x_row)

    # The cross-examination uses a deterministic class-score routine derived
    # from the same per-feature profile the LLM saw at fit time. This keeps
    # Round 2 reactions framework-imposed (and free of extra LLM calls).
    def class_scores(self, x_row: np.ndarray) -> Dict[str, float]:
        z = (x_row - self.mu) / self.sigma
        scores = {c: 0.0 for c in self.class_names}
        for j in self.selected:
            margin = abs(z[j])
            conf = _sigmoid(self.cfg.confidence_beta * margin * (0.5 + self.sep_norm[j]))
            obj = self.high_class[j] if z[j] >= 0 else self.low_class[j]
            scores[obj] += conf
        return scores


def build_agent(modality_name: str, cfg: AgentConfig, typed: bool) -> Agent:
    """Factory selecting the agent backend from config."""
    if cfg.type == "statistical":
        return StatisticalAgent(modality_name, cfg, typed=typed)
    if cfg.type == "llm":
        return LLMAgent(modality_name, cfg, typed=typed)
    raise ValueError(f"Unknown agent type '{cfg.type}'.")
