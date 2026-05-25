"""
download_pkt.py
===============
Download PheKnowLator v3.0.2 build files from Zenodo and derive the
processed NodeLabels CSV required by build_property_graph.py.

Zenodo record: https://zenodo.org/records/10056202
(Instance-based, Inverse Relations, OWLNETS, v3.0.2 - November 2021)

Files downloaded
----------------
Required by the pipeline
  PKT.nt.tar.gz                          -- RDF N-Triples (semantic backbone)
  PKT_NodeLabels.txt.tar.gz              -- raw node labels (tab-separated)

Optional / supporting (uncomment in PKT_OPTIONAL_FILES to download)
  PKT_NetworkxMultiDiGraph.gpickle.tar.gz
  PKT_decoding_dict.pkl.tar.gz
  node_metadata_dict.pkl.tar.gz
  Master_Edge_List_Dict.json.tar.gz
  PKT_Triples_Identifiers.txt.tar.gz
  PKT_Triples_Integers.txt.tar.gz
  PKT_Triples_Integer_Identifier_Map.json.tar.gz
  ontology_source_list.txt.zip
  edge_source_list.txt.zip
  downloaded_build_metadata.txt.zip

Derived output (generated locally, NOT downloaded)
  PKT_NodeLabels_with_metadata_v3.0.2.csv
      Built from PKT_NodeLabels.txt.tar.gz + LabelClassDataset_metadata.csv
      (the metadata CSV is curated manually and ships with the repository under
       data/pkt/builds/v3.0.2/LabelClassDataset_metadata.csv)

Usage
-----
    python scripts/download_pkt.py                  # download + process
    python scripts/download_pkt.py --skip-download  # process only (files already present)
    python scripts/download_pkt.py --download-only  # download only, skip processing

Run from the project root or from the scripts/ directory.
"""

import argparse
import os
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "data" / "pkt" / "builds" / "v3.0.2"

METADATA_CSV = BUILD_DIR / "LabelClassDataset_metadata.csv"   # curated, ships in repo

# ---------------------------------------------------------------------------
# Zenodo configuration
# ---------------------------------------------------------------------------

ZENODO_RECORD = "10056202"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Original Zenodo filename prefix (files are renamed to PKT_* on download)
_PKT_PREFIX = "PheKnowLator_v3.0.2_full_instance_inverseRelations_OWLNETS_INSTANCE_purified"

# Files required by the pipeline  {zenodo_filename: local_filename}
PKT_REQUIRED_FILES = {
    f"{_PKT_PREFIX}.nt.tar.gz":             "PKT.nt.tar.gz",
    f"{_PKT_PREFIX}_NodeLabels.txt.tar.gz": "PKT_NodeLabels.txt.tar.gz",
}

# Optional supporting files (uncomment as needed)
PKT_OPTIONAL_FILES = {
    # f"{_PKT_PREFIX}_NetworkxMultiDiGraph.gpickle.tar.gz": "PKT_NetworkxMultiDiGraph.gpickle.tar.gz",
    # f"{_PKT_PREFIX}_decoding_dict.pkl.tar.gz":            "PKT_decoding_dict.pkl.tar.gz",
    # "node_metadata_dict.pkl.tar.gz":                      "node_metadata_dict.pkl.tar.gz",
    # "Master_Edge_List_Dict.json.tar.gz":                  "Master_Edge_List_Dict.json.tar.gz",
    # f"{_PKT_PREFIX}_Triples_Identifiers.txt.tar.gz":      "PKT_Triples_Identifiers.txt.tar.gz",
    # f"{_PKT_PREFIX}_Triples_Integers.txt.tar.gz":         "PKT_Triples_Integers.txt.tar.gz",
    # f"{_PKT_PREFIX}_Triples_Integer_Identifier_Map.json.tar.gz": "PKT_Triples_Integer_Identifier_Map.json.tar.gz",
    # "ontology_source_list.txt.zip":                       "ontology_source_list.txt.zip",
    # "edge_source_list.txt.zip":                           "edge_source_list.txt.zip",
    # "downloaded_build_metadata.txt.zip":                  "downloaded_build_metadata.txt.zip",
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    print(f"  Downloading {dest.name} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=dest.name, ncols=80, leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved  {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")


def download_pkt_files(files: dict, force: bool = False) -> None:
    """Download a dict of {zenodo_name: local_name} into BUILD_DIR."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for zenodo_name, local_name in files.items():
        dest = BUILD_DIR / local_name
        if dest.exists() and not force:
            print(f"  Already present, skipping: {local_name}")
            continue
        url = f"{ZENODO_BASE_URL}/{zenodo_name}?download=1"
        _download_file(url, dest)


# ---------------------------------------------------------------------------
# NodeLabels processing
# (mirrors the logic in scripts/kg_utils/pkt_nodelabel-processing.py)
# ---------------------------------------------------------------------------

def _read_node_labels(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if not m.name.startswith(".")]
        extracted = tar.extractfile(members[0])
        return pd.read_csv(extracted, sep="\t", dtype=str)


def _extract_entity_from_uri(uri: str) -> str:
    if pd.isna(uri):
        return ""
    part = uri.rstrip(">").split("/")[-1]
    return part.split("=")[-1]


def _extract_class_code(entity: str) -> str:
    s = re.sub(r"(?<!\d)\d+$", "", entity).rstrip("_-")
    for prefix in ("PR_", "CHR_", "GNO_"):
        if s.startswith(prefix):
            return prefix.rstrip("_")
    return s.split("#")[0]


def _process_node_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "integer_id" in df.columns:
        df["integer_id"] = df["integer_id"].astype(str)
    df["entity"] = df["entity_uri"].apply(_extract_entity_from_uri) if "entity_uri" in df.columns else ""
    df["class_code"] = df["entity"].apply(_extract_class_code)
    df["class_code"] = (
        df["class_code"]
        .replace("UMLS_C", "UMLS")
        .replace("22-rdf-syntax-ns", "RDF")
        .replace("", "EntrezID")
        .replace("rs", "dbSNP")
    )
    if "entity_class" in df.columns:
        df = df.drop(columns=["entity_class"])
    return df


def build_node_labels_csv(force: bool = False) -> Path:
    """
    Read PKT_NodeLabels.txt.tar.gz, derive class_code and entity columns,
    merge with the curated LabelClassDataset_metadata.csv, and write
    PKT_NodeLabels_with_metadata_v3.0.2.csv to BUILD_DIR.

    Returns the path to the output CSV.
    """
    out_path = BUILD_DIR / "PKT_NodeLabels_with_metadata_v3.0.2.csv"
    if out_path.exists() and not force:
        print(f"  Already present, skipping: {out_path.name}")
        return out_path

    tar_path = BUILD_DIR / "PKT_NodeLabels.txt.tar.gz"
    if not tar_path.exists():
        sys.exit(f"[ERROR] Missing {tar_path}. Run without --skip-download first.")

    if not METADATA_CSV.exists():
        sys.exit(
            f"[ERROR] Curated metadata file not found: {METADATA_CSV}\n"
            "This file ships with the repository under data/pkt/builds/v3.0.2/."
        )

    print(f"  Reading {tar_path.name} ...")
    df = _read_node_labels(tar_path)
    print(f"  Read {len(df):,} rows. Processing ...")
    df = _process_node_labels(df)

    meta = pd.read_csv(METADATA_CSV)
    # drop duplicate (class_code, bioentity_type) rows that appear in the curated file
    meta = meta.drop_duplicates(subset=["class_code"])
    df = df.merge(meta, on="class_code", how="left")

    df.to_csv(out_path, index=False)
    print(f"  Written {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download PheKnowLator v3.0.2 build files and derive NodeLabels CSV."
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading from Zenodo (files must already be present)."
    )
    p.add_argument(
        "--download-only", action="store_true",
        help="Download files but skip NodeLabels processing."
    )
    p.add_argument(
        "--include-optional", action="store_true",
        help="Also download optional supporting files (NetworkX graph, dicts, etc.)."
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download and reprocess even if output files already exist."
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("download_pkt — PheKnowLator v3.0.2")
    print(f"Zenodo record : {ZENODO_RECORD}")
    print(f"Build dir     : {BUILD_DIR}")
    print("=" * 60)

    # --- Download ---
    if not args.skip_download:
        files_to_download = dict(PKT_REQUIRED_FILES)
        if args.include_optional:
            files_to_download.update(PKT_OPTIONAL_FILES)
        print(f"\n[1/2] Downloading {len(files_to_download)} file(s) from Zenodo ...")
        download_pkt_files(files_to_download, force=args.force)
    else:
        print("\n[1/2] Download skipped (--skip-download).")

    # --- Process NodeLabels ---
    if not args.download_only:
        print("\n[2/2] Building PKT_NodeLabels_with_metadata_v3.0.2.csv ...")
        out = build_node_labels_csv(force=args.force)
        print(f"\nDone. NodeLabels CSV: {out}")
    else:
        print("\n[2/2] Processing skipped (--download-only).")

    print("\n[OK] download_pkt finished.")


if __name__ == "__main__":
    main()