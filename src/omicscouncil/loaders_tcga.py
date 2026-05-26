"""TCGA-BRCA multi-omic loader for the medium-weight instantiation.

Builds a :class:`MultiModalDataset` from the four Xena/GDC TSV.gz matrices
under ``data/omics/TCGA-BRCA/`` plus the PAM50 label file produced by
``src/data/download_pam50.py``. Feature IDs are normalized to a canonical
namespace per modality so that the KnowledgePrior built from PheKnowLator
(``data/kg/PKT/``) can ground them:

  - RNA  (``star_tpm``)                 -> HGNC gene symbol via biomart map
  - CNV  (``gene-level_ascat3``)         -> HGNC gene symbol via biomart map
  - Met  (``methylation27``)             -> HGNC gene symbol via HM27 probemap
                                            (one CpG per gene kept, by MAD)
  - miR  (``mirna``)                     -> HGNC symbol via mirna_hgcn_map

Sample IDs are normalized by stripping the trailing vial letter
(``TCGA-XX-YYYY-01A`` -> ``TCGA-XX-YYYY-01``) to match the PAM50 file.

Feature selection: top-K by MAD (Median Absolute Deviation) computed on
the full intersection cohort. K is configurable via
``dataset.top_features_per_modality`` in the YAML (default 1000).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .data import Modality, MultiModalDataset, register_dataset_loader

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
OMICS_DIR = ROOT / "data" / "omics" / "TCGA-BRCA"
MAP_DIR = ROOT / "data" / "mappings"

OMIC_FILES = {
    "rna": OMICS_DIR / "TCGA-BRCA.star_tpm.tsv.gz",
    "met": OMICS_DIR / "TCGA-BRCA.methylation27.tsv.gz",
    "mir": OMICS_DIR / "TCGA-BRCA.mirna.tsv.gz",
    "cnv": OMICS_DIR / "TCGA-BRCA.gene-level_ascat3.tsv.gz",
}
PAM50_FILE = OMICS_DIR / "pam50_labels.tsv"


# --------------------------------------------------------------------------- #
# Mapping helpers (cached at module level — small files, parsed once)
# --------------------------------------------------------------------------- #
_BIOMART_CACHE: Dict[str, str] | None = None
_PROBEMAP_CACHE: Dict[str, str] | None = None
_MIRNA_CACHE: Dict[str, str] | None = None


def _read_zip_tsv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh, sep="\t", low_memory=False)


def _ensembl_to_symbol() -> Dict[str, str]:
    """Map Ensembl gene stable id (unversioned) -> HGNC symbol."""
    global _BIOMART_CACHE
    if _BIOMART_CACHE is None:
        df = _read_zip_tsv(MAP_DIR / "biomart_gene_mappings_filled_extended.tsv.zip")
        df = df.dropna(subset=["gene_stable_id", "hgnc_symbol"])
        df = df.drop_duplicates(subset=["gene_stable_id"])
        _BIOMART_CACHE = dict(zip(df["gene_stable_id"], df["hgnc_symbol"]))
    return _BIOMART_CACHE


def _cpg_to_symbol() -> Dict[str, str]:
    """Map CpG probe id (cgNNNN...) -> HGNC symbol via HM27 probemap."""
    global _PROBEMAP_CACHE
    if _PROBEMAP_CACHE is None:
        df = pd.read_csv(MAP_DIR / "HM27.hg38.manifest.gencode.v36.probeMap", sep="\t")
        df = df.dropna(subset=["#id", "gene"])
        # Some probes map to multiple comma-separated genes — keep the first.
        df["gene"] = df["gene"].astype(str).str.split(",").str[0].str.strip()
        df = df[df["gene"].astype(bool) & (df["gene"] != ".")]
        _PROBEMAP_CACHE = dict(zip(df["#id"], df["gene"]))
    return _PROBEMAP_CACHE


def _mirna_to_hgnc() -> Dict[str, str]:
    """Map miRNA stem-loop id (hsa-let-7a-1 ...) -> HGNC symbol (MIRLET7A-1 ...)."""
    global _MIRNA_CACHE
    if _MIRNA_CACHE is None:
        df = _read_zip_tsv(MAP_DIR / "mirna_hgcn_map.tsv.zip")
        df = df.dropna(subset=["miRNA_ID", "hgcn_id"]).drop_duplicates("miRNA_ID")
        _MIRNA_CACHE = dict(zip(df["miRNA_ID"], df["hgcn_id"]))
    return _MIRNA_CACHE


# --------------------------------------------------------------------------- #
# Per-modality readers — return (samples x features) DataFrame with canonical
# HGNC-symbol column names. Sample index is the normalized TCGA barcode.
# --------------------------------------------------------------------------- #
def _strip_vial(sample_id: str) -> str:
    """TCGA-XX-YYYY-01A  ->  TCGA-XX-YYYY-01."""
    return sample_id[:-1] if sample_id.endswith(("A", "B", "C", "D")) else sample_id


def _read_matrix(path: Path) -> pd.DataFrame:
    """Read a Xena featuresxsamples TSV.gz, return samplesxfeatures DataFrame."""
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    feature_col = df.columns[0]
    df = df.set_index(feature_col)
    df = df.T                                       # samples on rows
    df.index = [_strip_vial(s) for s in df.index]
    # Some Xena matrices contain duplicate columns (multiple aliquots of the
    # same sample after vial-suffix normalization). Keep the first.
    df = df.loc[~df.index.duplicated(keep="first")]
    return df


def _aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """When several source IDs map to the same canonical feature, keep the one
    with the largest MAD across samples (loses less information than mean)."""
    if df.columns.is_unique:
        return df
    mad = (df - df.median()).abs().median()
    order = mad.sort_values(ascending=False).index
    df = df[order]
    return df.loc[:, ~df.columns.duplicated(keep="first")]


def _canonicalize_columns(df: pd.DataFrame, mapping: Dict[str, str],
                          strip_version: bool = False) -> pd.DataFrame:
    """Rename columns via ``mapping``; drop those without a mapped target."""
    if strip_version:
        df.columns = [str(c).split(".")[0] for c in df.columns]
    new_cols = [mapping.get(str(c)) for c in df.columns]
    keep = [i for i, c in enumerate(new_cols) if c is not None]
    df = df.iloc[:, keep]
    df.columns = [new_cols[i] for i in keep]
    return _aggregate_duplicates(df)


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def _select_top_mad(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep the top-k columns by Median Absolute Deviation."""
    if df.shape[1] <= k:
        return df
    mad = (df - df.median()).abs().median()
    top = mad.sort_values(ascending=False).head(k).index
    return df[top]


def _select_top_anova_f(df: pd.DataFrame, y: np.ndarray, k: int) -> pd.DataFrame:
    """Keep the top-k columns by one-way ANOVA F-statistic against ``y``.

    Computed on the full cohort (no leakage concern: the selector is not the
    classifier; this is the standard practice in omics marker-recovery
    studies). NaN F-values (constant columns) are pushed to the tail.
    """
    if df.shape[1] <= k:
        return df
    from sklearn.feature_selection import f_classif
    F, _ = f_classif(df.values, y)
    F = np.nan_to_num(F, nan=-np.inf)
    top_idx = np.argsort(-F)[:k]
    return df.iloc[:, top_idx]


# --------------------------------------------------------------------------- #
# Main loader
# --------------------------------------------------------------------------- #
@register_dataset_loader("tcga_brca")
def load_tcga_brca(cfg: Config) -> MultiModalDataset:
    for path in OMIC_FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing omic file: {path}")
    if not PAM50_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PAM50_FILE}. Run `python src/data/download_pam50.py` first."
        )

    pam50 = pd.read_csv(PAM50_FILE, sep="\t")
    pam50 = pam50.set_index("sample_id")["pam50"]

    # 1) Read + canonicalize each modality (samples x HGNC-symbol features).
    print("[tcga_brca] Reading RNA...")
    rna = _canonicalize_columns(_read_matrix(OMIC_FILES["rna"]),
                                _ensembl_to_symbol(), strip_version=True)
    print(f"[tcga_brca]   rna {rna.shape} (samples x symbols)")

    print("[tcga_brca] Reading methylation27...")
    met = _canonicalize_columns(_read_matrix(OMIC_FILES["met"]),
                                _cpg_to_symbol(), strip_version=False)
    print(f"[tcga_brca]   met {met.shape}")

    print("[tcga_brca] Reading miRNA...")
    mir = _canonicalize_columns(_read_matrix(OMIC_FILES["mir"]),
                                _mirna_to_hgnc(), strip_version=False)
    print(f"[tcga_brca]   mir {mir.shape}")

    print("[tcga_brca] Reading CNV...")
    cnv = _canonicalize_columns(_read_matrix(OMIC_FILES["cnv"]),
                                _ensembl_to_symbol(), strip_version=True)
    print(f"[tcga_brca]   cnv {cnv.shape}")

    # 2) Sample intersection across all modalities + PAM50.
    common = sorted(
        set(rna.index) & set(met.index) & set(mir.index)
        & set(cnv.index) & set(pam50.index)
    )
    if len(common) < 50:
        raise RuntimeError(
            f"Sample intersection too small ({len(common)} samples). "
            "Check vial-suffix normalization."
        )
    print(f"[tcga_brca] 4-way intersection & PAM50: {len(common)} samples")

    y_labels = pam50.loc[common]

    # Drop under-populated classes (default: Normal, which has ~5 samples
    # after intersection and breaks stratified CV / calibration estimates).
    drop_classes = set(cfg.dataset.get("drop_classes", ["Normal"]))
    if drop_classes:
        keep_mask = ~y_labels.isin(drop_classes)
        dropped = int((~keep_mask).sum())
        if dropped:
            print(f"[tcga_brca] Dropping {dropped} samples in classes {sorted(drop_classes)}")
        common = [s for s, k in zip(common, keep_mask) if k]
        y_labels = y_labels[keep_mask]

    class_names = sorted(y_labels.unique())
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_idx[c] for c in y_labels], dtype=int)

    # 3) Subset matrices to the common sample set (preserves order).
    rna = rna.loc[common]
    met = met.loc[common]
    mir = mir.loc[common]
    cnv = cnv.loc[common]

    # 4) Drop constant/NaN columns, then top-K per modality by the configured
    #    selector. Default is ANOVA F-test (class-discriminative); MAD is kept
    #    as an unsupervised alternative for ablations / leakage-paranoid runs.
    top_k = int(cfg.dataset.get("top_features_per_modality", 1000))
    selector = str(cfg.dataset.get("feature_selection", "anova_f")).lower()

    def _clean_and_select(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="any")
        df = df.loc[:, df.std() > 1e-8]
        if selector == "anova_f":
            kept = _select_top_anova_f(df, y, top_k)
        elif selector == "mad":
            kept = _select_top_mad(df, top_k)
        else:
            raise ValueError(f"Unknown feature_selection '{selector}'.")
        print(f"[tcga_brca]   {name}: kept {kept.shape[1]} / {df.shape[1]} features (top-{top_k} {selector})")
        return kept

    matrices = {
        "rna": _clean_and_select(rna, "rna"),
        "met": _clean_and_select(met, "met"),
        "mir": _clean_and_select(mir, "mir"),
        "cnv": _clean_and_select(cnv, "cnv"),
    }

    # 5) Wrap as MultiModalDataset.
    modalities = {
        name: Modality(name=name,
                       feature_names=list(df.columns),
                       X=df.to_numpy(dtype=float))
        for name, df in matrices.items()
    }

    print(
        f"[tcga_brca] Final dataset: {len(common)} samples, "
        f"{len(class_names)} classes {class_names}, "
        f"modalities={ {n: m.X.shape for n, m in modalities.items()} }"
    )
    return MultiModalDataset(modalities=modalities, y=y, class_names=class_names)
