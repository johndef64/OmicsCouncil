"""
omics_utils.py
==============
Read / manipulation helpers for the locally downloaded TCGA / TARGET omic
matrices. Decoupled from download_omics.py (which only fetches files).

Typical usage:

    from omics_utils import read_data_type, PROBEMAP_PATHS, OMICS_PATHS

    df = read_data_type("mirna")                 # default study (TCGA-BRCA)
    df = read_data_type("mirna", "TCGA-LUAD")    # explicit study

By default, paths are anchored to data/omics/TCGA-BRCA/ so that scripts
that worked with the previous omics_connector.py keep working.
"""

import os
from pathlib import Path

import pandas as pd

# Default study used by helpers that don't take a `study` argument explicitly.
DEFAULT_STUDY = "TCGA-BRCA"

# Data types this module knows how to locate (must match download_omics.DATA_TYPES).
DATA_TYPES = [
    "gene-level_ascat3",
    "methylation27",
    "mirna",
    "star_tpm",
    "protein",
    "clinical",
    "survival",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OMICS_ROOT = PROJECT_ROOT / "data" / "omics"
MAPS_ROOT = OMICS_ROOT / "maps"

PROBEMAP_PATHS = [
    str(MAPS_ROOT / "gencode.v36.annotation.gtf.gene.probemap"),
    str(MAPS_ROOT / "HM450.hg38.manifest.gencode.v36.probeMap"),
    str(MAPS_ROOT / "HM27.hg38.manifest.gencode.v36.probeMap"),
]


def omics_paths(study: str = DEFAULT_STUDY) -> dict:
    """Return {data_type: tsv.gz path} for a given study."""
    root = OMICS_ROOT / study
    return {dt: str(root / f"{study}.{dt}.tsv.gz") for dt in DATA_TYPES}


def metadata_paths(study: str = DEFAULT_STUDY) -> dict:
    """Return {data_type: tsv.json path} for a given study."""
    root = OMICS_ROOT / study
    return {dt: str(root / f"{study}.{dt}.tsv.json") for dt in DATA_TYPES}


# Pre-built dict for the default study, kept for backward compatibility with
# code that did `from omics_connector import OMICS_PATHS`.
OMICS_PATHS = omics_paths(DEFAULT_STUDY)
METADATA_PATHS = metadata_paths(DEFAULT_STUDY)


def read_data_type(data_type: str, study: str = DEFAULT_STUDY):
    """Read one omic data type for the given study. Returns a DataFrame or None."""
    file_path = omics_paths(study).get(data_type)
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False)
    print(f"File not found for data type {data_type}: {file_path}")
    return None


def print_df_heads(study: str = DEFAULT_STUDY):
    """Preview the first 5 cols × 3 rows of every omic matrix for a study."""
    for dt, path in omics_paths(study).items():
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False, nrows=3)
            print(f"\nData Type: {dt}")
            print(df.iloc[:3, :5])
        else:
            print(f"\nData Type: {dt} - File not found: {path}")


def make_mirna_hgcn_map(study: str = DEFAULT_STUDY):
    """Build a miRNA-ID -> HGNC-symbol mapping table from the study's mirna matrix."""
    mirna_df = read_data_type("mirna", study)
    if mirna_df is None:
        return None
    mirna_df["hgcn_id"] = (
        mirna_df["miRNA_ID"]
        .str.replace("hsa-mir-", "MIR")
        .str.replace("hsa-let-", "MIRLET")
        .str.upper()
    )
    maps = mirna_df[["hgcn_id", "miRNA_ID"]]
    MAPS_ROOT.mkdir(parents=True, exist_ok=True)
    maps.to_csv(MAPS_ROOT / "mirna_hgcn_map.tsv", sep="\t", index=False)
    return maps


def make_clinical_sub_df(clinical_df, suffix):
    """Slice clinical_df columns by a suffix; prepend 'sample' as the first column."""
    sub_df = clinical_df[[c for c in clinical_df.columns if c.endswith(suffix)]]
    sub_df.insert(0, "sample", clinical_df["sample"])
    sub_df.columns = [c.replace(f".{suffix}", "") for c in sub_df.columns]
    return sub_df
