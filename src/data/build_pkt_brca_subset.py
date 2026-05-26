"""Build a PheKnowLator subset induced on the TCGA-BRCA selected features.

Pipeline:
  1. Materialize the TCGA-BRCA MultiModalDataset to get the union of HGNC
     symbols selected across the 4 modalities (~4000 unique).
  2. Stream `data/kg/PKT/nodes.json` and build a mapping
        node_uri  ->  canonical HGNC symbol  (matched on label/synonym)
     restricted to bioentity_type in {gene, protein, rna} so the subgraph is
     interpretable (we drop SNPs, chemicals, etc. — they are not in our
     feature space).
  3. Stream `data/kg/PKT/edges.zip!edges.json` and keep:
       (a) "induced" edges where BOTH endpoints map to a selected symbol  [DEFAULT]
       (b) optionally also "1-hop" edges where exactly one endpoint maps
           (controlled by --n-hops 1).
  4. Emit `data/kg/PKT/brca_subset.json` as a list of typed triples
       {subject, relation, object, weight}
     with subject/object = canonical HGNC symbol, relation = predicate_label,
     weight = 1.0 (per-edge confidence is not available in PKT; the arbiter
     uses tercile thresholds so uniform weights still partition into E1/E2/E3).

The output is a drop-in for `KnowledgePrior.from_file` — same interface used
by the lightweight derive_from_data path.

Note: PKT predicate vocabulary contains both directions (e.g. "located_in"
vs "location_of"); we keep them as separate edges. The OmicsCouncil arbiter
looks claims up by exact (subject, relation, object) so direction matters.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterator, Set

# Make the package importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

NODES_FILE = ROOT / "data" / "kg" / "PKT" / "nodes.json"
EDGES_ZIP = ROOT / "data" / "kg" / "PKT" / "edges.zip"
OUT_FILE = ROOT / "data" / "kg" / "PKT" / "brca_subset.json"

# Node types we accept for symbol grounding. Variants/chemicals/cells are
# outside our feature space.
ACCEPTED_BIOENTITY = {"gene", "protein", "rna"}

# For 1-hop expansion: bioentity_types we accept as "anchor neighbours" of a
# selected gene. These are the only annotations the Opt-C arbiter needs to
# stratify claim grades (pathway / GO membership for E1, phenotype / disease
# for richer interpretability traces). Chemicals, anatomy, organisms are
# excluded as noisy / off-topic for the PAM50 task.
ONE_HOP_BIOENTITY = {"pathway", "go", "disease", "phenotype"}


def _parse_label_symbol(label: str) -> str | None:
    """Extract the leading HGNC-like token from a PKT label.

    PKT gene labels look like ``"ACAD10 (human)"`` and miRNA stem-loop
    transcripts like ``"MIRLET7A1"``. Returns the first whitespace-separated
    token (upper-cased) or ``None`` if it looks like a sentence.
    """
    if not label:
        return None
    token = label.split()[0].strip()
    # Reject sentence-like labels ("acyl-CoA dehydrogenase ...") whose first
    # word is mixed case / lower-case.
    if not token or len(token) > 30:
        return None
    if token.isupper() or (token[0].isupper() and any(ch.isdigit() for ch in token)):
        return token
    # Pure upper tokens or symbol-with-dash
    if all(ch.isupper() or ch.isdigit() or ch in "-." for ch in token):
        return token
    return None


def _candidate_symbols_from_node(node: dict) -> Set[str]:
    """Symbols the node may be addressed by (label + synonyms).

    Synonyms in PKT are pipe-separated. We upper-case and strip parens.
    """
    out: Set[str] = set()
    lbl_sym = _parse_label_symbol(node.get("label", ""))
    if lbl_sym:
        out.add(lbl_sym.upper())
    syn = node.get("synonym", "") or ""
    for s in syn.split("|"):
        s = s.strip()
        if not s:
            continue
        # Drop parens/qualifiers e.g. "ACAD-10 (human)" -> "ACAD-10"
        s = s.split("(")[0].strip()
        if 1 <= len(s) <= 30:
            out.add(s.upper())
    return out


def _stream_json_array(path: Path) -> Iterator[dict]:
    """Yield records from a JSON array file by repeatedly decoding chunks.

    Falls back to a one-shot json.load for files small enough to fit; for the
    large PKT files (~500MB) uses ijson if available, else streams via
    raw_decode.
    """
    try:
        import ijson  # type: ignore
        with open(path, "rb") as fh:
            for obj in ijson.items(fh, "item"):
                yield obj
        return
    except ImportError:
        pass

    # Fallback: incremental raw_decode (slower, no extra deps).
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fh:
        buf = fh.read(1 << 20)
        # Skip leading whitespace + '['
        i = 0
        while i < len(buf) and buf[i] in " \t\r\n":
            i += 1
        if i < len(buf) and buf[i] == "[":
            i += 1
        while True:
            # Skip separators
            while i < len(buf) and buf[i] in " ,\t\r\n":
                i += 1
            if i < len(buf) and buf[i] == "]":
                return
            # Refill buffer if low
            if len(buf) - i < (1 << 18):
                more = fh.read(1 << 20)
                if more:
                    buf = buf[i:] + more
                    i = 0
            try:
                obj, end = decoder.raw_decode(buf, i)
            except json.JSONDecodeError:
                more = fh.read(1 << 20)
                if not more:
                    return
                buf = buf[i:] + more
                i = 0
                continue
            yield obj
            i = end


def _stream_zip_json(zip_path: Path, member: str = "edges.json") -> Iterator[dict]:
    try:
        import ijson  # type: ignore
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as fh:
                for obj in ijson.items(fh, "item"):
                    yield obj
        return
    except ImportError:
        pass

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            text = fh.read().decode("utf-8")
    for obj in json.loads(text):
        yield obj


def get_selected_symbols(top_k: int) -> Set[str]:
    """Re-run the loader to fetch the 4 x top_k HGNC feature symbols."""
    from omicscouncil.config import (
        AgentConfig, ClassifierConfig, Config, CouncilConfig, KnowledgeConfig,
    )
    from omicscouncil.data import build_multimodal_dataset

    cfg = Config(
        domain="tcga_brca", seed=42, test_size=0.30,
        dataset={"loader": "tcga_brca", "top_features_per_modality": top_k},
        modalities={"rna": {}, "met": {}, "mir": {}, "cnv": {}},
        relations=["elevated_in", "reduced_in"],
        agent=AgentConfig(), council=CouncilConfig(),
        knowledge=KnowledgeConfig(), classifier=ClassifierConfig(),
    )
    ds = build_multimodal_dataset(cfg)
    syms: Set[str] = set()
    for m in ds.modalities.values():
        syms.update(str(s).upper() for s in m.feature_names)
    return syms


def build_node_index(symbols: Set[str]) -> tuple[Dict[str, str], Dict[str, dict]]:
    """Stream PKT nodes and return (uri_to_sym, uri_to_anchor_node).

    ``uri_to_sym`` maps gene/protein/rna URIs to a canonical HGNC symbol
    selected from our feature space. ``uri_to_anchor_node`` retains, for the
    1-hop expansion, the metadata of nodes whose ``bioentity_type`` is in
    :data:`ONE_HOP_BIOENTITY` (pathway/GO/disease/phenotype) so we can later
    label the "object" side of cross-type edges with something readable.
    """
    print(f"[pkt-subset] Streaming nodes from {NODES_FILE}", flush=True)
    uri_to_sym: Dict[str, str] = {}
    uri_to_anchor: Dict[str, dict] = {}
    n_seen = 0
    n_hit = 0
    for node in _stream_json_array(NODES_FILE):
        n_seen += 1
        if n_seen % 100_000 == 0:
            print(f"[pkt-subset]   ...scanned {n_seen:,} nodes, {n_hit:,} symbol hits", flush=True)
        btype = node.get("bioentity_type")
        uri = node.get("uri")
        if not uri:
            continue
        if btype in ACCEPTED_BIOENTITY:
            cands = _candidate_symbols_from_node(node)
            match = cands & symbols
            if match and uri not in uri_to_sym:
                uri_to_sym[uri] = sorted(match)[0]
                n_hit += 1
        elif btype in ONE_HOP_BIOENTITY:
            # Keep a lightweight anchor record for readable object labels.
            uri_to_anchor[uri] = {
                "label": node.get("label") or node.get("entity_id") or uri.rsplit("/", 1)[-1],
                "class_code": node.get("class_code", ""),
                "bioentity_type": btype,
            }
    print(
        f"[pkt-subset] Scanned {n_seen:,} nodes; matched {n_hit:,} URIs to "
        f"{len(set(uri_to_sym.values())):,} unique symbols; "
        f"{len(uri_to_anchor):,} anchor nodes ({sorted(ONE_HOP_BIOENTITY)})"
    )
    return uri_to_sym, uri_to_anchor


def build_edge_subset(uri_to_sym: Dict[str, str],
                      uri_to_anchor: Dict[str, dict],
                      n_hops: int) -> list:
    """Stream PKT edges, keep those induced by ``uri_to_sym`` (and optionally
    1-hop neighbours restricted to :data:`ONE_HOP_BIOENTITY`).
    """
    print(f"[pkt-subset] Streaming edges from {EDGES_ZIP} (n_hops={n_hops})", flush=True)
    triples: list = []
    seen_keys = set()
    n_seen = 0
    n_induced = 0
    n_onehop = 0
    for edge in _stream_zip_json(EDGES_ZIP):
        n_seen += 1
        if n_seen % 500_000 == 0:
            print(
                f"[pkt-subset]   ...scanned {n_seen:,} edges, kept {len(triples):,} "
                f"(induced {n_induced:,} | 1-hop {n_onehop:,})",
                flush=True,
            )
        s_uri = edge.get("source_uri")
        t_uri = edge.get("target_uri")
        s_sym = uri_to_sym.get(s_uri)
        t_sym = uri_to_sym.get(t_uri)
        anchor_obj = None  # filled when only one side is a gene symbol

        if s_sym and t_sym:
            n_induced += 1
            obj_btype = "gene"
        elif n_hops >= 1 and (s_sym or t_sym):
            # Only accept 1-hop when the *other* endpoint is in our anchor
            # node set (pathway / GO / disease / phenotype). This prunes out
            # noisy hops to chemicals/anatomy/organism.
            anchor_uri = t_uri if s_sym else s_uri
            anchor = uri_to_anchor.get(anchor_uri)
            if anchor is None:
                continue
            anchor_obj = anchor
            obj_btype = anchor["bioentity_type"]
            # Canonicalize so that subject = gene symbol, object = anchor label.
            if not s_sym:
                # Swap so the symbol is always the subject (for arbiter lookups).
                s_sym = t_sym
                t_sym = anchor["label"]
            else:
                t_sym = anchor["label"]
            n_onehop += 1
        else:
            continue

        rel = (edge.get("predicate_label") or "related_to").strip().replace(" ", "_")
        key = (s_sym, rel, t_sym)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        triples.append({
            "subject": s_sym,
            "relation": rel,
            "object": t_sym,
            "object_type": obj_btype,
            "weight": 1.0,
        })
    print(
        f"[pkt-subset] Scanned {n_seen:,} edges; kept {len(triples):,} unique triples "
        f"(induced {n_induced:,}, 1-hop {n_onehop:,})"
    )
    return triples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=1000,
                    help="features per modality (must match the run config)")
    ap.add_argument("--n-hops", type=int, default=0,
                    help="0 = induced subgraph only; 1 = include 1-hop neighbours")
    ap.add_argument("--out", type=str, default=str(OUT_FILE),
                    help="output JSON path")
    args = ap.parse_args()

    symbols = get_selected_symbols(args.top_k)
    print(f"[pkt-subset] Selected feature symbols: {len(symbols):,}")
    uri_to_sym, uri_to_anchor = build_node_index(symbols)
    triples = build_edge_subset(uri_to_sym, uri_to_anchor, n_hops=args.n_hops)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(triples, fh)
    print(f"[pkt-subset] Wrote {out_path} ({len(triples):,} triples)")


if __name__ == "__main__":
    main()
