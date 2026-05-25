"""
download_omics.py
=================
Download TCGA / TARGET multi-omic data files from the UCSC Xena GDC Hub
(https://gdc-hub.s3.us-east-1.amazonaws.com/download/) into data/omics/.

Each study × data_type produces two files:
    {study}.{data_type}.tsv.gz   -- the matrix
    {study}.{data_type}.tsv.json -- the metadata sidecar

Usage examples
--------------
    # Single study (default: TCGA-BRCA) -- quick test
    python scripts/download_omics.py

    # Arbitrary single study
    python scripts/download_omics.py --studies TCGA-LUAD

    # Multiple specific studies
    python scripts/download_omics.py --studies TCGA-BRCA TCGA-LUAD TARGET-AML

    # Full TCGA cohort
    python scripts/download_omics.py --cohort tcga

    # Full TARGET cohort
    python scripts/download_omics.py --cohort target

    # Both TCGA and TARGET
    python scripts/download_omics.py --cohort all

    # Restrict the data types
    python scripts/download_omics.py --studies TCGA-BRCA --data-types mirna protein

    # Include probemaps and / or PanCan auxiliary files
    python scripts/download_omics.py --include-probemaps --include-pancan
"""

import argparse
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OMICS_DIR = PROJECT_ROOT / "data" / "omics"
MAPS_DIR = OMICS_DIR / "maps"
PANCAN_DIR = OMICS_DIR / "PANCAN"

# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

TCGA_STUDIES = [
    "TCGA-LAML", "TCGA-ACC", "TCGA-CHOL", "TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC",
    "TCGA-COAD", "TCGA-UCEC", "TCGA-ESCA", "TCGA-GBM",  "TCGA-HNSC", "TCGA-KICH",
    "TCGA-KIRC", "TCGA-KIRP", "TCGA-DLBC", "TCGA-LIHC", "TCGA-LGG",  "TCGA-LUAD",
    "TCGA-LUSC", "TCGA-SKCM", "TCGA-MESO", "TCGA-UVM",  "TCGA-OV",   "TCGA-PAAD",
    "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ", "TCGA-SARC", "TCGA-STAD", "TCGA-TGCT",
    "TCGA-THYM", "TCGA-THCA", "TCGA-UCS",
]

TARGET_STUDIES = [
    "TARGET-ALL-P1", "TARGET-ALL-P2", "TARGET-ALL-P3", "TARGET-AML",
    "TARGET-CCSK",   "TARGET-NBL",    "TARGET-OS",     "TARGET-RT", "TARGET-WT",
]

# Data types actually used by the pipeline (see use_case_*.py)
DATA_TYPES = [
    "gene-level_ascat3",   # Copy Number (Gene Level)
    "methylation27",       # DNA Methylation (Illumina HM27)
    "mirna",               # miRNA Expression
    "star_tpm",            # Gene Expression (STAR - TPM)
    "protein",             # Protein Expression (RPPA, TCGA only)
    "clinical",            # Clinical Data
    "survival",            # Survival Data
]

# Full catalogue (for reference / advanced use; not downloaded by default)
DATA_TYPES_ALL = [
    "gene-level_ascat2", "gene-level_ascat3", "allele_cnv_ascat3",
    "methylation27", "methylation450",
    "somaticmutation_wxs",
    "mirna",
    "star_counts", "star_fpkm", "star_tpm",
    "protein",
    "clinical", "survival",
]

GDC_ROOT_URL = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"

PROBEMAP_URLS = [
    f"{GDC_ROOT_URL}gencode.v36.annotation.gtf.gene.probemap",
    f"{GDC_ROOT_URL}HM450.hg38.manifest.gencode.v36.probeMap",
    f"{GDC_ROOT_URL}HM27.hg38.manifest.gencode.v36.probeMap",
]

PANCAN_URLS = [
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGASubtype.20170308.tsv.gz",
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
]

# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download `url` to `dest`. Returns True on success, False if HTTP != 200."""
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False


def download_study(study: str, data_types: list, force: bool = False) -> dict:
    """
    Download all requested data types for one study into data/omics/{study}/.
    Returns a summary dict {downloaded: [...], missing: [...], skipped: [...]}.
    """
    out_dir = OMICS_DIR / study
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nStudy: {study}\n{'='*60}")
    summary = {"downloaded": [], "missing": [], "skipped": []}

    for dt in data_types:
        data_url     = f"{GDC_ROOT_URL}{study}.{dt}.tsv.gz"
        metadata_url = f"{GDC_ROOT_URL}{study}.{dt}.tsv.json"
        data_file    = out_dir / f"{study}.{dt}.tsv.gz"
        meta_file    = out_dir / f"{study}.{dt}.tsv.json"

        if data_file.exists() and not force:
            print(f"  [skip] {dt}  (already present)")
            summary["skipped"].append(dt)
            continue

        # HEAD probe -- many study/data-type combinations don't exist on the hub
        try:
            head = requests.head(data_url, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"  [fail] {dt}  ({e})")
            summary["missing"].append(dt)
            continue

        if head.status_code != 200:
            print(f"  [miss] {dt}  (HTTP {head.status_code})")
            summary["missing"].append(dt)
            continue

        print(f"  [get ] {dt} ...", end=" ", flush=True)
        ok = _stream_download(data_url, data_file)
        if not ok:
            print("failed")
            summary["missing"].append(dt)
            continue
        _stream_download(metadata_url, meta_file)  # best-effort
        size_mb = data_file.stat().st_size / 1e6
        print(f"ok ({size_mb:.1f} MB)")
        summary["downloaded"].append(dt)

    print(
        f"  -> downloaded={len(summary['downloaded'])} "
        f"skipped={len(summary['skipped'])} "
        f"missing={len(summary['missing'])}"
    )
    return summary


def download_url_list(urls: list, out_dir: Path, label: str, force: bool = False):
    """Download a flat list of URLs into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = out_dir / name
        if dest.exists() and not force:
            print(f"  [skip] {name}  (already present)")
            continue
        print(f"  [get ] {name} ...", end=" ", flush=True)
        ok = _stream_download(url, dest)
        print("ok" if ok else "failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download TCGA / TARGET multi-omic data from the UCSC Xena GDC Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--studies", nargs="+", metavar="STUDY",
        help="One or more study IDs (e.g. TCGA-BRCA TCGA-LUAD TARGET-AML).",
    )
    group.add_argument(
        "--cohort", choices=["tcga", "target", "all"],
        help="Download an entire cohort. 'all' = TCGA + TARGET.",
    )
    p.add_argument(
        "--data-types", nargs="+", metavar="TYPE", default=DATA_TYPES,
        help=f"Data types to fetch per study (default: {' '.join(DATA_TYPES)}).",
    )
    p.add_argument(
        "--include-probemaps", action="store_true",
        help=f"Also download probemap files into {MAPS_DIR.relative_to(PROJECT_ROOT)}.",
    )
    p.add_argument(
        "--include-pancan", action="store_true",
        help=f"Also download TCGA PanCan auxiliary files into {PANCAN_DIR.relative_to(PROJECT_ROOT)}.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download files that are already present locally.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List available studies and data types, then exit.",
    )
    return p.parse_args()


def resolve_studies(args) -> list:
    if args.studies:
        return args.studies
    if args.cohort == "tcga":
        return TCGA_STUDIES
    if args.cohort == "target":
        return TARGET_STUDIES
    if args.cohort == "all":
        return TCGA_STUDIES + TARGET_STUDIES
    # default: BRCA test run
    return ["TCGA-BRCA"]


def main():
    args = parse_args()

    if args.list:
        print("TCGA studies:")
        for s in TCGA_STUDIES: print(f"  {s}")
        print("\nTARGET studies:")
        for s in TARGET_STUDIES: print(f"  {s}")
        print("\nDefault data types:")
        for d in DATA_TYPES: print(f"  {d}")
        print("\nFull data type catalogue:")
        for d in DATA_TYPES_ALL: print(f"  {d}")
        return

    studies = resolve_studies(args)
    if not args.studies and not args.cohort:
        print("[info] No --studies or --cohort provided; defaulting to TCGA-BRCA (test run).")
        print("       Use --cohort tcga|target|all  or  --studies <ID> ...  for more.")

    print("=" * 60)
    print(f"download_omics — target: {len(studies)} study(ies)")
    print(f"data types    : {', '.join(args.data_types)}")
    print(f"output root   : {OMICS_DIR}")
    print("=" * 60)

    overall = {"downloaded": 0, "skipped": 0, "missing": 0}
    for study in studies:
        s = download_study(study, args.data_types, force=args.force)
        overall["downloaded"] += len(s["downloaded"])
        overall["skipped"]    += len(s["skipped"])
        overall["missing"]    += len(s["missing"])

    if args.include_probemaps:
        download_url_list(PROBEMAP_URLS, MAPS_DIR, "Probemap files", force=args.force)
    if args.include_pancan:
        download_url_list(PANCAN_URLS, PANCAN_DIR, "PanCan auxiliary files", force=args.force)

    print("\n" + "=" * 60)
    print(f"[OK] download_omics finished. "
          f"downloaded={overall['downloaded']}  "
          f"skipped={overall['skipped']}  "
          f"missing={overall['missing']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
