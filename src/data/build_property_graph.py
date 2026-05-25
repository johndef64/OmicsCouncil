"""
build_property_graph.py
=======================
Convert the PheKnowLator RDF N-Triples dump into an ArangoDB-ready property
graph (nodes.json + edges.json) by enriching every entity with the metadata
contained in PKT_NodeLabels_with_metadata_v3.0.2.csv.

Inputs  (data/pkt/builds/v3.0.2/, produced by scripts/download_pkt.py):
    PKT.nt.tar.gz
    PKT_NodeLabels_with_metadata_v3.0.2.csv

Outputs (by default in data/pkt/builds/v3.0.2/property_graph/):
    nodes.json   -- one document per unique RDF entity
    edges.json   -- one document per triple
    metadata_lookup.zip  (cache of the URI -> metadata dict)

Usage
-----
    # Full build with default paths
    python scripts/build_property_graph.py

    # Quick sample run (10k triples, useful for debugging)
    python scripts/build_property_graph.py --sample 10000

    # Custom paths
    python scripts/build_property_graph.py \\
        --pkt-build-dir data/pkt/builds/v3.0.2 \\
        --output-dir data/pkt/builds/v3.0.2/property_graph

    # Rebuild the metadata lookup even if its cache exists
    python scripts/build_property_graph.py --force-metadata
"""

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from REHMED.src.data.pkt_utils import read_tar_rdf

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BUILD_DIR = PROJECT_ROOT / "data" / "pkt" / "builds" / "v3.0.2"
DEFAULT_OUTPUT_DIR = DEFAULT_BUILD_DIR / "property_graph"


# ---------------------------------------------------------------------------
# Metadata lookup
# ---------------------------------------------------------------------------

def create_metadata_lookup(nodelabels_df: pd.DataFrame) -> dict:
    """Build a URI -> metadata-dict map from the NodeLabels CSV."""
    print("Creating metadata lookup from NodeLabels ...")
    nodelabels_df["clean_uri"] = nodelabels_df["entity_uri"].str.strip("<>")
    lookup = {}
    for _, row in nodelabels_df.iterrows():
        lookup[row["clean_uri"]] = {
            "entity_type_cat": row["entity_type"],   # NODES or RELATIONS
            "integer_id":      row["integer_id"],
            "label":           row["label"]                  if pd.notna(row["label"])                  else "",
            "description":     row["description/definition"] if pd.notna(row["description/definition"]) else "",
            "synonym":         row["synonym"]                if pd.notna(row["synonym"])                else "",
            "entity":          row["entity"]                 if pd.notna(row["entity"])                 else "",
            "class_code":      row["class_code"]             if pd.notna(row["class_code"])             else "",
            "bioentity_type":  row["bioentity_type"]         if pd.notna(row["bioentity_type"])         else "",
            "source":          row["source"]                 if pd.notna(row["source"])                 else "",
            "source_type":     row["source_type"]            if pd.notna(row["source_type"])            else "",
        }
    print(f"  metadata lookup contains {len(lookup):,} entries")
    return lookup


def get_entity_metadata(uri: str, metadata_lookup: dict) -> dict:
    """Return metadata for `uri`; fall back to URI-derived defaults if missing."""
    clean_uri = uri.strip("<>")
    parsed = urlparse(clean_uri)
    namespace = parsed.netloc or "unknown"

    if clean_uri in metadata_lookup:
        m = metadata_lookup[clean_uri]
        return {
            "uri":            clean_uri,
            "namespace":      namespace,
            "entity_id":      m["entity"],
            "class_code":     m["class_code"],
            "label":          m["label"],
            "bioentity_type": m["bioentity_type"],
            "description":    m["description"],
            "synonym":        m["synonym"],
            "source":         m["source"],
            "source_type":    m["source_type"],
            "integer_id":     m["integer_id"],
        }
    entity_id = clean_uri.split("/")[-1]
    return {
        "uri": clean_uri, "namespace": namespace, "entity_id": entity_id,
        "class_code": "unknown", "label": entity_id, "bioentity_type": "unknown",
        "description": "", "synonym": "", "source": "unknown", "source_type": "unknown",
        "integer_id": None,
    }


def load_or_build_metadata_lookup(node_label_file: Path, cache_dir: Path,
                                  force: bool = False) -> dict:
    """Reuse the zipped metadata_lookup cache if present, otherwise build and cache it."""
    cache_zip = cache_dir / "metadata_lookup.zip"
    if cache_zip.exists() and not force:
        print(f"Loading cached metadata lookup from {cache_zip} ...")
        with zipfile.ZipFile(cache_zip, "r") as zf, zf.open("metadata_lookup.json") as fh:
            return json.load(fh)

    print(f"Loading NodeLabels from {node_label_file} ...")
    nodelabels = pd.read_csv(node_label_file)
    print(f"  read {len(nodelabels):,} rows")
    lookup = create_metadata_lookup(nodelabels)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_json = cache_dir / "metadata_lookup.json"
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(lookup, fh, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(cache_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_json, arcname="metadata_lookup.json")
    tmp_json.unlink()
    print(f"  cached lookup to {cache_zip}")
    return lookup


# ---------------------------------------------------------------------------
# Triple -> property graph conversion
# ---------------------------------------------------------------------------

# Characters that need escaping when building an ArangoDB _key
_KEY_REPLACE = str.maketrans({c: "_" for c in ":/?=#@."})


def _safe_key(entity_id: str) -> str:
    return str(entity_id).translate(_KEY_REPLACE)


def convert_rdf_to_property_graph(df: pd.DataFrame, metadata_lookup: dict):
    """Convert an RDF triples DataFrame into parallel (nodes, edges) lists."""
    nodes, edges = {}, []
    print("Converting RDF triples to property graph ...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="triples", ncols=80):
        subject, predicate, obj = row["subject"], row["predicate"], row["object"]

        for uri in (subject, obj):
            if uri in nodes:
                continue
            meta = get_entity_metadata(uri, metadata_lookup)
            nodes[uri] = {
                "_key":           _safe_key(meta["entity_id"]),
                "uri":            meta["uri"],
                "namespace":      meta["namespace"],
                "entity_id":      meta["entity_id"],
                "class_code":     meta["class_code"],
                "label":          meta["label"],
                "bioentity_type": meta["bioentity_type"],
                "description":    meta["description"],
                "synonym":        meta["synonym"],
                "source":         meta["source"],
                "source_type":    meta["source_type"],
                "integer_id":     meta["integer_id"],
            }

        pred_meta = get_entity_metadata(predicate, metadata_lookup)
        edges.append({
            "edge_id":                  f"edge_{idx}",
            "source_uri":               subject.strip("<>"),
            "target_uri":               obj.strip("<>"),
            "predicate_uri":            pred_meta["uri"],
            "predicate_label":          pred_meta["label"],
            "predicate_class_code":     pred_meta["class_code"],
            "predicate_bioentity_type": pred_meta["bioentity_type"],
            "predicate_source":         pred_meta["source"],
        })

    return list(nodes.values()), edges


# ---------------------------------------------------------------------------
# Export + reporting
# ---------------------------------------------------------------------------

def export_to_json_for_arangodb(nodes, edges, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "nodes.json", "w", encoding="utf-8") as fh:
        json.dump(nodes, fh, indent=2, ensure_ascii=False)
    with open(output_dir / "edges.json", "w", encoding="utf-8") as fh:
        json.dump(edges, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote nodes.json + edges.json to {output_dir}")


def print_distributions(nodes, edges):
    node_types, edge_types = defaultdict(int), defaultdict(int)
    for n in nodes: node_types[n["bioentity_type"]] += 1
    for e in edges: edge_types[e["predicate_label"]] += 1

    print(f"\nConversion complete: {len(nodes):,} nodes, {len(edges):,} edges")
    print("\nTop 20 bioentity types:")
    for k, v in sorted(node_types.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}")
    print("\nTop 20 predicate labels:")
    for k, v in sorted(edge_types.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert the PKT RDF dump into an ArangoDB-ready property graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )
    p.add_argument("--pkt-build-dir", type=Path, default=DEFAULT_BUILD_DIR,
                   help="Directory containing PKT.nt.tar.gz + NodeLabels CSV.")
    p.add_argument("--pkt-file", type=Path, default=None,
                   help="Override path to PKT.nt.tar.gz (default: <pkt-build-dir>/PKT.nt.tar.gz).")
    p.add_argument("--node-label-file", type=Path, default=None,
                   help="Override path to PKT_NodeLabels_with_metadata_v3.0.2.csv.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write nodes.json + edges.json.")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Convert only a random sample of N triples (output goes to <output-dir>/sample_<timestamp>/).")
    p.add_argument("--force-metadata", action="store_true",
                   help="Rebuild the metadata lookup even if its cache zip exists.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used for --sample (default: 42).")
    return p.parse_args()


def main():
    args = parse_args()

    pkt_file        = args.pkt_file        or (args.pkt_build_dir / "PKT.nt.tar.gz")
    node_label_file = args.node_label_file or (args.pkt_build_dir / "PKT_NodeLabels_with_metadata_v3.0.2.csv")

    for required in (pkt_file, node_label_file):
        if not required.exists():
            sys.exit(f"[ERROR] Required input not found: {required}\n"
                     f"Run scripts/download_pkt.py first.")

    output_dir = args.output_dir
    if args.sample:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / f"sample_{ts}"
        print(f"[sample mode] using up to {args.sample:,} triples, output -> {output_dir}")

    print("=" * 60)
    print("build_property_graph")
    print(f"  PKT triples : {pkt_file}")
    print(f"  NodeLabels  : {node_label_file}")
    print(f"  output dir  : {output_dir}")
    print("=" * 60)

    print(f"\nLoading RDF triples from {pkt_file.name} ...")
    pkt_kg = read_tar_rdf(str(pkt_file))
    print(f"  loaded {len(pkt_kg):,} triples")

    metadata_lookup = load_or_build_metadata_lookup(
        node_label_file, args.pkt_build_dir, force=args.force_metadata
    )

    if args.sample:
        n = min(args.sample, len(pkt_kg))
        df = pkt_kg.sample(n, random_state=args.seed)
        print(f"\nSampling {n:,} triples (seed={args.seed}) ...")
    else:
        df = pkt_kg

    nodes, edges = convert_rdf_to_property_graph(df, metadata_lookup)
    print_distributions(nodes, edges)
    export_to_json_for_arangodb(nodes, edges, output_dir)

    print("\n" + "=" * 60)
    print("[OK] property graph build completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
