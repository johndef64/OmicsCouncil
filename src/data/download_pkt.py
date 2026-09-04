"""
download_pkt.py
===============
Fetch the PheKnowLator property graph used as the knowledge prior and place it
where ``build_pkt_brca_subset.py`` expects it.

Source
------
The ``PKT/`` directory of the KG-TransomicNet dataset repository on HuggingFace,
which distributes a property-graph rendering of the PheKnowLator v3.0.2
OWL-NETS build (Callahan et al.).

    https://huggingface.co/datasets/johndef64/KG-TransomicNet

We do not rebuild that rendering from the raw OWL-NETS release: that
construction belongs to the upstream project and is not reproduced here. The
downloaded nodes already carry the metadata the pipeline needs
(``bioentity_type``, ``class_code``, ``label``), so no separate metadata file
is required.

What this script does
---------------------
    PKT/nodes.zip  (~65 MB)   ->  data/kg/PKT/nodes.json  (unzipped, ~483 MB)
    PKT/edges.zip  (~225 MB)  ->  data/kg/PKT/edges.zip   (left zipped)

The asymmetry is deliberate, not an oversight: ``build_pkt_brca_subset.py``
streams ``nodes.json`` in the clear and reads ``edges.json`` from inside
``edges.zip``. Unzipping edges.zip breaks the next step.

Usage
-----
    python src/data/download_pkt.py
    python src/data/download_pkt.py --force       # re-download and re-extract
    python src/data/download_pkt.py --keep-zip    # keep nodes.zip after extraction
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — note parents[1]: this file lives in src/data/, the project root is
# two levels up. Must agree with build_pkt_brca_subset.py, which reads
# data/kg/PKT/.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
KG_DIR = PROJECT_ROOT / "data" / "kg" / "PKT"

# ---------------------------------------------------------------------------
# HuggingFace source
# ---------------------------------------------------------------------------

HF_REPO = "johndef64/KG-TransomicNet"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/PKT"

# {remote name: (local name, extract_to or None)}
PKT_FILES = {
    "nodes.zip": ("nodes.zip", "nodes.json"),   # extracted, then optionally removed
    "edges.zip": ("edges.zip", None),           # left as-is, read as a zip
}


def _download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  Already present, skipping: {dest.name} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Downloading {dest.name} from {HF_REPO} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=dest.name, ncols=80, leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved  {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")


def _extract(zip_path: Path, member: str, force: bool = False) -> Path:
    out = zip_path.parent / member
    if out.exists() and not force:
        print(f"  Already extracted, skipping: {member} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
        return out
    print(f"  Extracting {member} from {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if member not in names:
            sys.exit(f"[ERROR] {member} not found in {zip_path.name}: {names[:5]}")
        zf.extract(member, path=zip_path.parent)
    print(f"  Written {member}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Download the PheKnowLator property graph (KG-TransomicNet "
                    "PKT/) into data/kg/PKT/."
    )
    p.add_argument("--force", action="store_true",
                   help="Re-download and re-extract even if files are present.")
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep nodes.zip after extracting nodes.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 68)
    print("download_pkt — PheKnowLator property graph")
    print(f"Source    : {HF_REPO} (PKT/)")
    print(f"Target dir: {KG_DIR}")
    print("=" * 68)

    KG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] Downloading {len(PKT_FILES)} file(s) ...")
    for remote, (local, _) in PKT_FILES.items():
        _download(f"{HF_BASE}/{remote}", KG_DIR / local, force=args.force)

    print("\n[2/2] Extracting ...")
    for _, (local, member) in PKT_FILES.items():
        if member is None:
            print(f"  {local} is read as a zip by build_pkt_brca_subset.py; "
                  f"leaving it compressed.")
            continue
        zip_path = KG_DIR / local
        _extract(zip_path, member, force=args.force)
        if not args.keep_zip:
            zip_path.unlink()
            print(f"  Removed {local} (use --keep-zip to retain it)")

    print("\n[OK] download_pkt finished. Next:")
    print("     python src/data/build_pkt_brca_subset.py --top-k 3000 --n-hops 1")


if __name__ == "__main__":
    main()
