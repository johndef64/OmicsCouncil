"""Shared helpers for the experiment scripts: import bootstrap, data loading,
pretty printing, and a simple feature-concatenation baseline.

Running ``python experiments/run_*.py`` works without installing the package:
we add ``src`` to ``sys.path`` here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- import bootstrap ------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Ensure GROQ_API_KEY is set from src/api_keys.json if the LLMAgent is used.
_keys_path = os.path.join(_ROOT, "src", "api_keys.json")
if os.path.exists(_keys_path) and not os.environ.get("GROQ_API_KEY"):
    import json as _json
    _keys = _json.load(open(_keys_path))
    if _keys.get("groq"):
        os.environ["GROQ_API_KEY"] = _keys["groq"]

from omicscouncil.config import Config, load_config            # noqa: E402
from omicscouncil.data import (                                # noqa: E402
    MultiModalDataset, build_multimodal_dataset, train_test_split_dataset,
)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "breast_cancer.yaml")
RESULTS_DIR = os.path.join(_ROOT, "results")


def load_split(config_path: str) -> Tuple[Config, MultiModalDataset, MultiModalDataset]:
    cfg = load_config(config_path)
    ds = build_multimodal_dataset(cfg)
    train, test = train_test_split_dataset(ds, cfg.test_size, cfg.seed)
    return cfg, train, test


def save_json(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def print_table(title: str, rows: List[Dict[str, object]], cols: List[str]) -> None:
    print(f"\n{title}")
    widths = {c: max(len(c), *(len(_fmt(r.get(c, ''))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols))


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class ConcatBaseline:
    """Early-integration baseline: concatenate all modalities, scale, fit a
    logistic regression. Missing modalities are mean-imputed at test time, which
    is the standard fallback OmicsCouncil avoids by design."""

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def _concat(self, ds: MultiModalDataset, active: Optional[List[str]] = None,
                impute: bool = False) -> np.ndarray:
        active_set = set(active) if active is not None else set(ds.modality_names)
        blocks = []
        for m in ds.modality_names:
            X = ds.modalities[m].X
            if m not in active_set and impute:
                X = np.tile(self._means[m], (X.shape[0], 1))   # mean imputation
            blocks.append(X)
        return np.hstack(blocks)

    def fit(self, train: MultiModalDataset) -> "ConcatBaseline":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self._means = {m: train.modalities[m].X.mean(axis=0) for m in train.modality_names}
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=self.max_iter))
        self.clf.fit(self._concat(train), train.y)
        return self

    def predict(self, ds: MultiModalDataset, active: Optional[List[str]] = None) -> np.ndarray:
        return self.clf.predict(self._concat(ds, active=active, impute=True))

    def predict_proba(self, ds: MultiModalDataset, active: Optional[List[str]] = None) -> np.ndarray:
        return self.clf.predict_proba(self._concat(ds, active=active, impute=True))
