"""Dataset loading and slicing into heterogeneous modalities.

The framework consumes a :class:`MultiModalDataset`: a set of named modalities,
each a dense feature matrix, plus a shared label vector. New domains plug in by
registering a loader with :func:`register_loader` (or by adding one here) that
returns the raw feature matrix and names; the generic
:func:`build_multimodal_dataset` then splits features into modalities according
to the config's ``feature_filter`` tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import Config


@dataclass
class Modality:
    name: str
    feature_names: List[str]
    X: np.ndarray              # (n_samples, n_features_in_modality)


@dataclass
class MultiModalDataset:
    modalities: Dict[str, Modality]
    y: np.ndarray              # (n_samples,) integer labels
    class_names: List[str]

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def modality_names(self) -> List[str]:
        return list(self.modalities.keys())

    def subset(self, idx: np.ndarray) -> "MultiModalDataset":
        return MultiModalDataset(
            modalities={
                m: Modality(mod.name, mod.feature_names, mod.X[idx])
                for m, mod in self.modalities.items()
            },
            y=self.y[idx],
            class_names=self.class_names,
        )


# --------------------------------------------------------------------------- #
# Two loader registries:
#   * _LOADERS         — legacy "flat X + feature_filter" loaders (Wisconsin).
#   * _DATASET_LOADERS — modern loaders that build the per-modality matrices
#                        themselves (used for true multi-omic settings where
#                        each modality has a distinct feature space and sample
#                        intersection has to be computed up front).
# --------------------------------------------------------------------------- #
LoaderFn = Callable[[], Tuple[np.ndarray, List[str], np.ndarray, List[str]]]
_LOADERS: Dict[str, LoaderFn] = {}

DatasetLoaderFn = Callable[["Config"], "MultiModalDataset"]
_DATASET_LOADERS: Dict[str, DatasetLoaderFn] = {}


def register_loader(name: str) -> Callable[[LoaderFn], LoaderFn]:
    def deco(fn: LoaderFn) -> LoaderFn:
        _LOADERS[name] = fn
        return fn
    return deco


def register_dataset_loader(name: str) -> Callable[[DatasetLoaderFn], DatasetLoaderFn]:
    def deco(fn: DatasetLoaderFn) -> DatasetLoaderFn:
        _DATASET_LOADERS[name] = fn
        return fn
    return deco


@register_loader("sklearn_breast_cancer")
def _load_breast_cancer() -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Breast Cancer Wisconsin (Diagnostic). Ships with scikit-learn; no download.

    569 samples, 30 features, binary target (malignant / benign).
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return (
        np.asarray(data.data, dtype=float),
        [str(n) for n in data.feature_names],
        np.asarray(data.target, dtype=int),
        [str(c) for c in data.target_names],
    )


def build_multimodal_dataset(cfg: Config) -> MultiModalDataset:
    """Build a :class:`MultiModalDataset` from the config.

    Two paths are supported:

    1. **Dataset loader** (preferred for true multi-omic): the loader returns
       a fully assembled :class:`MultiModalDataset` (per-modality matrices,
       sample intersection, label vector) and the config's ``modalities``
       block is used only to validate the modality names.
    2. **Flat loader** (legacy lightweight path): the loader returns a single
       feature matrix; modalities are then carved out by ``feature_filter``.
    """
    loader_name = cfg.dataset["loader"]

    # Import dataset-loader modules for their side-effect registrations.
    from . import loaders_tcga  # noqa: F401

    if loader_name in _DATASET_LOADERS:
        ds = _DATASET_LOADERS[loader_name](cfg)
        declared = set(cfg.modalities)
        produced = set(ds.modality_names)
        if declared and declared != produced:
            raise ValueError(
                f"Loader '{loader_name}' produced modalities {sorted(produced)} "
                f"but config declares {sorted(declared)}."
            )
        return ds

    if loader_name not in _LOADERS:
        raise KeyError(
            f"Unknown loader '{loader_name}'. Registered flat: {sorted(_LOADERS)}; "
            f"registered dataset: {sorted(_DATASET_LOADERS)}"
        )
    X, feature_names, y, class_names = _LOADERS[loader_name]()

    modalities: Dict[str, Modality] = {}
    for mod_name, mod_spec in cfg.modalities.items():
        token = mod_spec["feature_filter"].lower()
        cols = [i for i, fn in enumerate(feature_names) if token in fn.lower()]
        if not cols:
            raise ValueError(
                f"Modality '{mod_name}' filter '{token}' matched no features."
            )
        modalities[mod_name] = Modality(
            name=mod_name,
            feature_names=[feature_names[i] for i in cols],
            X=X[:, cols],
        )

    return MultiModalDataset(modalities=modalities, y=y, class_names=class_names)


def train_test_split_dataset(
    ds: MultiModalDataset, test_size: float, seed: int
) -> Tuple[MultiModalDataset, MultiModalDataset]:
    """Stratified split that keeps every modality aligned to the same samples."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(ds.n_samples)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=ds.y
    )
    return ds.subset(train_idx), ds.subset(test_idx)
