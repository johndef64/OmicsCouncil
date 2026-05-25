"""Typed configuration objects.

A :class:`Config` is a thin, validated view over the YAML domain file. Keeping
all domain-specific knobs here (and nowhere else) is what makes the framework
portable: a new domain is a new YAML file plus, optionally, a new data loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class AgentConfig:
    type: str = "statistical"
    top_features_per_agent: int = 5
    confidence_threshold: float = 0.55
    confidence_beta: float = 1.5


@dataclass
class CouncilConfig:
    enable_cross_examination: bool = True
    enable_arbiter: bool = True
    typed_claims: bool = True
    drop_grades: List[str] = field(default_factory=lambda: ["E3", "E4"])
    support_threshold: int = 1


@dataclass
class KnowledgeConfig:
    source: str = "derive_from_data"   # 'derive_from_data' | 'file'
    path: str | None = None


@dataclass
class ClassifierConfig:
    type: str = "logistic_regression"
    max_iter: int = 1000


@dataclass
class Config:
    domain: str
    seed: int
    test_size: float
    dataset: Dict[str, Any]
    modalities: Dict[str, Dict[str, Any]]
    relations: List[str]
    agent: AgentConfig
    council: CouncilConfig
    knowledge: KnowledgeConfig
    classifier: ClassifierConfig
    reference_markers: List[str] = field(default_factory=list)

    @property
    def modality_names(self) -> List[str]:
        return list(self.modalities.keys())


def load_config(path: str) -> Config:
    """Parse a YAML domain file into a validated :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return Config(
        domain=raw["domain"],
        seed=int(raw.get("seed", 42)),
        test_size=float(raw.get("test_size", 0.3)),
        dataset=raw["dataset"],
        modalities=raw["modalities"],
        relations=list(raw["relations"]),
        agent=AgentConfig(**raw.get("agent", {})),
        council=CouncilConfig(**raw.get("council", {})),
        knowledge=KnowledgeConfig(**raw.get("knowledge", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        reference_markers=list(raw.get("reference_markers", [])),
    )
