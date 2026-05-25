"""End-to-end OmicsCouncil pipeline.

Design choice (mirrored in the paper): we separate *representation generation*
— the agentic deliberation, which is unsupervised at inference time — from
*prediction* — a lightweight classifier trained on the resulting Z-sym vectors.
This keeps the trustworthy, interpretable part (the trace) decoupled from the
small supervised head.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .agents import Agent, build_agent
from .claims import DeliberationTrace
from .config import Config
from .council import Council
from .data import MultiModalDataset
from .knowledge import KnowledgePrior, build_knowledge_prior
from .representation import RepresentationSchema, build_schema


class OmicsCouncilPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.agents: Dict[str, Agent] = {}
        self.prior: Optional[KnowledgePrior] = None
        self.schema: Optional[RepresentationSchema] = None
        self.council: Optional[Council] = None
        self.clf = None
        self.class_names: List[str] = []

    # --------------------------------------------------------------------- fit
    def fit(self, train: MultiModalDataset) -> "OmicsCouncilPipeline":
        self.class_names = train.class_names

        # 1) Modality experts.
        for mod_name, mod in train.modalities.items():
            agent = build_agent(mod_name, self.cfg.agent, typed=self.cfg.council.typed_claims)
            agent.fit(mod, train.y, train.class_names)
            self.agents[mod_name] = agent

        # 2) Knowledge prior + council protocol.
        self.prior = build_knowledge_prior(self.cfg.knowledge, train)
        self.council = Council(self.agents, self.prior, self.cfg.council)

        # 3) Representation schema (fixed layout derived from the domain).
        self.schema = build_schema(self.cfg.relations, train.class_names, train.modality_names)

        # 4) Lightweight supervised head on Z-sym.
        X_sym, _ = self.transform(train)
        self.clf = self._build_classifier()
        self.clf.fit(X_sym, train.y)
        return self

    def _build_classifier(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if self.cfg.classifier.type == "logistic_regression":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=self.cfg.classifier.max_iter),
            )
        raise ValueError(f"Unknown classifier '{self.cfg.classifier.type}'.")

    # --------------------------------------------------------------- transform
    def transform(
        self, ds: MultiModalDataset, active_modalities: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[DeliberationTrace]]:
        """Run the council on every sample and encode each trace as Z-sym.

        ``active_modalities`` restricts which modalities are observed — used to
        simulate missing modalities for the robustness experiment.
        """
        assert self.council is not None and self.schema is not None and self.prior is not None
        active = active_modalities or ds.modality_names

        vectors, traces = [], []
        for i in range(ds.n_samples):
            rows = {m: ds.modalities[m].X[i] for m in active if m in ds.modalities}
            trace = self.council.deliberate(rows, sample_id=i)
            vectors.append(self.schema.encode(trace, self.prior))
            traces.append(trace)
        return np.vstack(vectors), traces

    # ----------------------------------------------------------------- predict
    def predict_proba(
        self, ds: MultiModalDataset, active_modalities: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[DeliberationTrace]]:
        X_sym, traces = self.transform(ds, active_modalities)
        return self.clf.predict_proba(X_sym), traces

    def predict(
        self, ds: MultiModalDataset, active_modalities: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[DeliberationTrace]]:
        X_sym, traces = self.transform(ds, active_modalities)
        return self.clf.predict(X_sym), traces
