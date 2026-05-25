"""
build_omics_collections.py
==========================
Build the ArangoDB-ready JSON collections from per-study TCGA / TARGET omic
files previously fetched by scripts/download_omics.py.

For each requested study, the script produces (under
data/arangodb_collections/<STUDY>/) the following JSON files:

  Semantic layer
    genes.json
    projects.json

  Metadata layer
    samples.json
    cases.json

  Quantitative index layer (one document per cohort × omic)
    gene_expression_index.json
    cnv_index.json
    mirna_index.json
    protein_index.json
    methylation_index.json

  Quantitative vector layer (one document per sample × omic)
    gene_expression_samples_<STUDY>.json
    cnv_samples_<STUDY>.json
    mirna_samples_<STUDY>.json
    protein_samples_<STUDY>.json
    methylation_samples_<STUDY>.json

Each omic step is only executed if both the input matrix and the relevant
mappings exist; missing inputs are skipped with a warning.

Usage examples (mirror the CLI of scripts/download_omics.py)
------------------------------------------------------------
    # Single study (default: TCGA-BRCA)
    python scripts/build_omics_collections.py

    # Explicit list of studies
    python scripts/build_omics_collections.py --studies TCGA-BRCA TCGA-LUAD

    # Full cohort
    python scripts/build_omics_collections.py --cohort tcga
    python scripts/build_omics_collections.py --cohort target
    python scripts/build_omics_collections.py --cohort all

    # Restrict the omic layers to build
    python scripts/build_omics_collections.py --layers gene_expression mirna

    # Skip the semantic / metadata layers (rebuild only quantitative ones)
    python scripts/build_omics_collections.py --no-semantic --no-metadata
"""

import argparse
import ast
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Reuse the study catalogues from download_omics so the two CLIs stay in sync.
from REHMED.src.data.download_omics import TCGA_STUDIES, TARGET_STUDIES, GDC_ROOT_URL, PROBEMAP_URLS

# ---------------------------------------------------------------------------
# Paths and logging
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OMICS_ROOT = PROJECT_ROOT / "data" / "omics"
MAPS_ROOT = PROJECT_ROOT / "data" / "mappings"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "arangodb_collections"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Quantitative omic layers that this script can build.
# Each entry maps the layer name -> (data_type used by download_omics, builder method).
QUANT_LAYERS = ["gene_expression", "cnv", "mirna", "protein", "methylation"]


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_omics_data(study: str, data_type: str) -> Optional[pd.DataFrame]:
    """Load one TCGA/TARGET omic matrix for a study, or return None if absent."""
    file_path = OMICS_ROOT / study / f"{study}.{data_type}.tsv.gz"
    if file_path.exists():
        logger.info(f"Loading {data_type} from {file_path.name}")
        return pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False)
    logger.warning(f"File not found: {file_path}")
    return None


# --- Auto-provisioning of mapping files -----------------------------------
# Two strategies:
#   * Probemap files (3 of them) are publicly hosted on the GDC hub; if they
#     are missing we download them on the fly.
#   * Heavy curated mappings (4 of them) are shipped in the repository as .zip
#     archives next to where the .tsv is expected; if only the .zip is present
#     we transparently extract it before reading.

# Probemap files: filename -> public URL (GDC hub).
_PROBEMAP_DOWNLOADS = {url.rsplit("/", 1)[-1]: url for url in PROBEMAP_URLS}

# Mapping files shipped compressed in the repository (the .zip is committed,
# the .tsv is gitignored and produced by unzipping the archive).
_ZIPPED_MAPPINGS = {
    "biomart_gene_mappings_filled_extended.tsv",
    "biomart_ensg_enst_mapping.tsv",
    "rsid_checkpoint.tsv",
    "mirna_hgcn_map.tsv",
}


def _download_to(path: Path, url: str):
    """Stream-download `url` to `path` (parent dirs created as needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  downloading {path.name} from {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                fh.write(chunk)
    logger.info(f"  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def _ensure_mapping_available(fname: str) -> Optional[Path]:
    """
    Make sure the mapping file `fname` exists under MAPS_ROOT.
    Returns the resolved path, or None if the file cannot be obtained.

    Resolution order:
      1. <MAPS_ROOT>/<fname>          -- use as-is
      2. <MAPS_ROOT>/<fname>.zip      -- unzip in place
      3. probemap: download from GDC hub
    """
    target = MAPS_ROOT / fname
    if target.exists():
        return target

    # Auto-unzip if a .zip ships in the repo (heavy curated mappings)
    zip_path = MAPS_ROOT / f"{fname}.zip"
    if zip_path.exists():
        logger.info(f"  extracting {zip_path.name} -> {target.name}")
        MAPS_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MAPS_ROOT)
        if target.exists():
            return target
        logger.warning(f"  unzipped {zip_path.name} but {fname} not found inside")

    # Auto-download for publicly hosted probemap files
    if fname in _PROBEMAP_DOWNLOADS:
        try:
            _download_to(target, _PROBEMAP_DOWNLOADS[fname])
            return target
        except Exception as exc:
            logger.warning(f"  download failed for {fname}: {exc}")

    return None


def load_mappings() -> Dict[str, pd.DataFrame]:
    """
    Load every curated mapping table required by the builders.

    Files are resolved transparently:
      * heavy curated mappings shipped as .zip are extracted on first use;
      * probemap files are downloaded from the GDC hub on first use.
    """
    files = {
        "biomart_genes": "biomart_gene_mappings_filled_extended.tsv",
        "hm27":          "HM27.hg38.manifest.gencode.v36.probeMap",
        "hm450":         "HM450.hg38.manifest.gencode.v36.probeMap",
        "mirna":         "mirna_hgcn_map.tsv",
        "rppa":          "TCPA_RPPA500_metadata_mapping.tsv",
        "rsid":          "rsid_checkpoint.tsv",
        "mondo":         "mondo_to_primarydisease.tsv",
        "ensg_enst":     "biomart_ensg_enst_mapping.tsv",
    }
    dfs = {}
    for name, fname in files.items():
        path = _ensure_mapping_available(fname)
        if path is None:
            logger.warning(f"Mapping file not available: {fname}")
            continue
        dfs[name] = pd.read_csv(path, sep="\t", dtype=str)
        logger.info(f"  loaded mapping '{name}'  shape={dfs[name].shape}")
    return dfs


def create_lookup_dicts(mapping_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Build per-key lookup dicts for fast ID translation."""
    logger.info("Creating lookup dictionaries ...")
    if "biomart_genes" not in mapping_dfs:
        raise SystemExit(
            "[ERROR] Required mapping 'biomart_gene_mappings_filled_extended.tsv' is "
            "not available.\n"
            "        Expected at: data/mappings/biomart_gene_mappings_filled_extended.tsv\n"
            "        or its compressed counterpart: "
            "data/mappings/biomart_gene_mappings_filled_extended.tsv.zip\n"
            "        (the .zip ships with the repository — make sure you ran `git pull`)."
        )
    bm = mapping_dfs["biomart_genes"]
    lookups = {
        "ensembl_to_entrez":  pd.Series(bm["entrez_id"].values,            index=bm["gene_stable_id"]).to_dict(),
        "ensembl_to_hgnc":    pd.Series(bm["hgnc_symbol"].values,          index=bm["gene_stable_id"]).to_dict(),
        "ensembl_to_uniprot": pd.Series(bm["uniprot_swissprot_id"].values, index=bm["gene_stable_id"]).to_dict(),
        "ensembl_to_mirbase": pd.Series(bm["mirbase_id"].values,           index=bm["gene_stable_id"]).to_dict(),
    }
    if "ensg_enst" in mapping_dfs:
        lookups["ensg_to_ensts"] = (
            mapping_dfs["ensg_enst"].groupby("gene_id")["transcript_id"].apply(list).to_dict()
        )
    logger.info(f"  created {len(lookups)} lookup dictionaries")
    return lookups


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

class CollectionBuilder:
    """Base class: holds shared context and writes JSONL collection files."""

    def __init__(self, study: str, mapping_dfs: Dict, lookups: Dict, output_dir: Path):
        self.study = study
        self.mapping_dfs = mapping_dfs
        self.lookups = lookups
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_collection(self, documents: List[Dict], collection_name: str):
        out = self.output_dir / f"{collection_name}.json"
        with open(out, "w", encoding="utf-8") as fh:
            for doc in documents:
                json.dump(doc, fh, ensure_ascii=False)
                fh.write("\n")
        logger.info(f"  -> wrote {len(documents):,} documents to {out.name}")


# Semantic layer (stable biological entities) -------------------------------

class SemanticLayerBuilder(CollectionBuilder):

    def build_gene_nodes(self) -> List[Dict]:
        logger.info("Building GENES semantic collection ...")
        bm = self.mapping_dfs["biomart_genes"]
        genes = []
        for _, row in tqdm(bm.iterrows(), total=len(bm), desc="genes"):
            transcript_ids = []
            raw_tids = row.get("transcript_id")
            if pd.notna(raw_tids):
                try:
                    transcript_ids = ast.literal_eval(raw_tids) if isinstance(raw_tids, str) else [raw_tids]
                except Exception:
                    transcript_ids = []
            genes.append({
                "_key":                   row["gene_stable_id"],
                "entity_type":            "gene",
                "bioentity_type":         "gene",
                "gene_stable_id":         row["gene_stable_id"],
                "gene_stable_id_version": row.get("gene_stable_id_version"),
                "hgnc_symbol":            row.get("hgnc_symbol")          if pd.notna(row.get("hgnc_symbol"))          else None,
                "entrez_id":              str(int(float(row["entrez_id"]))) if pd.notna(row.get("entrez_id")) else None,
                "uniprot_id":             row.get("uniprot_swissprot_id") if pd.notna(row.get("uniprot_swissprot_id")) else None,
                "mirbase_id":             row.get("mirbase_id")           if pd.notna(row.get("mirbase_id"))           else None,
                "gene_type":              row.get("gene_type"),
                "gene_description":       row.get("gene_description"),
                "chromosome":             row.get("chromosome"),
                "gene_start_bp":          row.get("gene_start_bp"),
                "gene_end_bp":            row.get("gene_end_bp"),
                "strand":                 row.get("strand"),
                "transcript_ids":         transcript_ids,
                "source":                 "Ensembl_BioMart",
                "source_version":         "GRCh38.v36",
            })
        return genes

    def build_project_node(self, clinical_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building PROJECT collection ...")
        return [{
            "_key":           clinical_df.loc[0, "project_id.project"],
            "name":           clinical_df.loc[0, "name.project"],
            "program":        clinical_df.loc[0, "name.program.project"],
            "primary_site":   clinical_df.loc[0, "primary_site.project"],
            "disease_types":  ast.literal_eval(clinical_df.loc[0, "disease_type.project"]),
            "entity_type":    "project",
            "n_cases":        int(clinical_df["submitter_id"].nunique()),
            "n_samples":      int(clinical_df["sample"].nunique()),
        }]


# Quantitative index layer (per-cohort gene/probe ordering) ----------------

class QuantitativeIndexBuilder(CollectionBuilder):

    def build_expression_index(self, star_tpm_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building EXPRESSION_INDEX collection ...")
        full_ids = star_tpm_df["Ensembl_ID"].values
        base_ids = [g.split(".")[0] for g in full_ids]

        gene_mappings = []
        for i, (full, base) in enumerate(zip(full_ids, base_ids)):
            entrez = self.lookups["ensembl_to_entrez"].get(base)
            hgnc   = self.lookups["ensembl_to_hgnc"].get(base)
            gene_mappings.append({
                "position":         i,
                "gene_id_ensembl":  full,
                "gene_id_base":     base,
                "gene_ref":         f"genes/{base}",
                "entrez_id":        str(int(float(entrez))) if pd.notna(entrez) else None,
                "hgnc_symbol":      hgnc if pd.notna(hgnc) else None,
            })
        return [{
            "_key":           f"expr_index_{self.study}",
            "cohort":         self.study,
            "data_type":      "gene_expression_index",
            "n_genes":        len(base_ids),
            "platform":       "STAR",
            "genome_version": "GRCh38.v36",
            "gene_mappings":  gene_mappings,
            "description":    "Gene position mapping for expression vectors in sample documents.",
        }]

    def build_cnv_index(self, cnv_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building CNV_INDEX collection ...")
        full_ids = cnv_df["Ensembl_ID"].values
        base_ids = [g.split(".")[0] for g in full_ids]

        gene_mappings = []
        for i, (full, base) in enumerate(zip(full_ids, base_ids)):
            entrez = self.lookups["ensembl_to_entrez"].get(base)
            ensts  = self.lookups.get("ensg_to_ensts", {}).get(base, [])
            gene_mappings.append({
                "position":         i,
                "gene_id_ensembl":  full,
                "gene_id_base":     base,
                "gene_ref":         f"genes/{base}",
                "entrez_id":        str(int(float(entrez))) if pd.notna(entrez) else None,
                "enst_ids":         ensts if ensts else None,
            })
        return [{
            "_key":            f"cnv_index_{self.study}",
            "cohort":          self.study,
            "data_type":       "cnv_index",
            "n_genes":         len(base_ids),
            "analysis_method": "ASCAT3",
            "gene_mappings":   gene_mappings,
            "description":     "Gene position mapping for CNV vectors in sample documents.",
        }]

    def build_methylation_index(self, methylation_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building METHYLATION_INDEX collection ...")
        probe_ids = methylation_df["Composite Element REF"].values
        hm27_map = self.mapping_dfs.get("hm27")

        # Pre-compute the reverse HGNC -> Ensembl lookup once.
        hgnc_to_ensembl = {v: k for k, v in self.lookups["ensembl_to_hgnc"].items() if pd.notna(v)}

        probe_mappings = []
        for i, probe_id in tqdm(enumerate(probe_ids), total=len(probe_ids), desc="probes"):
            match = hm27_map[hm27_map["#id"] == probe_id] if hm27_map is not None else None
            if match is not None and not match.empty:
                r = match.iloc[0]
                chrom       = r.get("chrom", "*")
                chrom_start = r.get("chromStart", -1)
                chrom_end   = r.get("chromEnd", -1)
                strand      = r.get("strand", None)
                gene_str    = r.get("gene", "")
                gene_symbols = [g.strip() for g in gene_str.split(",")] if pd.notna(gene_str) and gene_str else []
                gene_ids    = [hgnc_to_ensembl[s] for s in gene_symbols if s in hgnc_to_ensembl]
                probe_mappings.append({
                    "position":      i,
                    "probe_id":      probe_id,
                    "chromosome":    chrom if chrom != "*" else None,
                    "genomic_start": int(chrom_start) if chrom_start != -1 else None,
                    "genomic_end":   int(chrom_end)   if chrom_end   != -1 else None,
                    "strand":        strand if pd.notna(strand) else None,
                    "gene_symbols":  gene_symbols or None,
                    "gene_ids":      gene_ids or None,
                    "gene_refs":     [f"genes/{g}" for g in gene_ids] if gene_ids else None,
                })
            else:
                probe_mappings.append({
                    "position": i, "probe_id": probe_id, "chromosome": None,
                    "genomic_start": None, "genomic_end": None, "strand": None,
                    "gene_symbols": None, "gene_ids": None, "gene_refs": None,
                })

        mapped       = sum(1 for p in probe_mappings if p["chromosome"] is not None)
        gene_annot   = sum(1 for p in probe_mappings if p["gene_ids"])
        logger.info(f"  mapped {mapped}/{len(probe_ids)} probes; gene-annotated {gene_annot}/{len(probe_ids)}")

        return [{
            "_key":            f"methylation_index_{self.study}",
            "cohort":          self.study,
            "data_type":       "methylation_index",
            "n_probes":        len(probe_ids),
            "platform":        "Illumina HumanMethylation27",
            "genome_version":  "GRCh38",
            "probe_mappings":  probe_mappings,
            "description":     "CpG probe position mapping for methylation beta-value vectors.",
        }]

    def build_mirna_index(self, mirna_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building MIRNA_INDEX collection ...")
        mirna_ids  = mirna_df["miRNA_ID"].values
        mirna_hgcn = mirna_df["hgcn_id"].values if "hgcn_id" in mirna_df.columns else [None] * len(mirna_ids)
        mirna_mappings = [
            {
                "position":    i,
                "mirna_id":    mid,
                "mirbase_id":  mid,
                "hgnc_symbol": mirna_hgcn[i] if pd.notna(mirna_hgcn[i]) else None,
                "description": f"miRNA {mid}",
            }
            for i, mid in enumerate(mirna_ids)
        ]
        return [{
            "_key":            f"mirna_index_{self.study}",
            "cohort":          self.study,
            "data_type":       "mirna_index",
            "n_mirnas":        len(mirna_ids),
            "platform":        "Illumina",
            "mirna_mappings":  mirna_mappings,
            "description":     "miRNA position mapping for expression vectors.",
        }]

    def build_protein_index(self, protein_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building PROTEIN_INDEX collection ...")
        peptide_ids = protein_df["peptide_target"].values
        rppa_map = self.mapping_dfs.get("rppa")
        protein_mappings = []
        for i, pid in enumerate(peptide_ids):
            entrez_id = gene_symbol = protein_type = None
            if rppa_map is not None:
                match = rppa_map[rppa_map["RPPA_Protein_ID"] == pid]
                if not match.empty:
                    entrez_id    = match.iloc[0].get("Entrez_Gene_ID")
                    gene_symbol  = match.iloc[0].get("Gene_Symbol")
                    protein_type = match.iloc[0].get("Protein_Type")
            protein_mappings.append({
                "position":       i,
                "peptide_target": pid,
                "entrez_id":      entrez_id,
                "gene_symbol":    gene_symbol,
                "protein_type":   protein_type,
            })
        return [{
            "_key":              f"protein_index_{self.study}",
            "cohort":            self.study,
            "data_type":         "protein_index",
            "n_proteins":        len(peptide_ids),
            "platform":          "RPPA",
            "protein_mappings":  protein_mappings,
            "description":       "Protein/peptide position mapping for RPPA vectors.",
        }]


# Quantitative vector layer (one document per sample) ----------------------

class SampleCentricVectorBuilder(CollectionBuilder):

    def build_gene_expression_samples(self, star_tpm_df: pd.DataFrame,
                                      star_counts_df: Optional[pd.DataFrame] = None,
                                      star_fpkm_df:   Optional[pd.DataFrame] = None) -> List[Dict]:
        logger.info("Building GENE_EXPRESSION_SAMPLES collection ...")
        sample_ids = star_tpm_df.columns[1:].tolist()
        docs = []
        for sid in tqdm(sample_ids, desc="expression samples"):
            tpm_vec    = pd.to_numeric(star_tpm_df[sid],    errors="coerce").tolist()
            counts_vec = pd.to_numeric(star_counts_df[sid], errors="coerce").tolist() if star_counts_df is not None else None
            fpkm_vec   = pd.to_numeric(star_fpkm_df[sid],   errors="coerce").tolist() if star_fpkm_df   is not None else None
            docs.append({
                "_key":                 sid,
                "sample_id":            sid,
                "cohort":               self.study,
                "data_type":            "gene_expression_vector",
                "expression_index_ref": f"expr_index_{self.study}",
                "platform":             "STAR",
                "n_genes":              len(tpm_vec),
                "values_counts":        counts_vec,
                "values_fpkm":          fpkm_vec,
                "values_tpm":           tpm_vec,
                "normalization":        "TPM/FPKM/counts",
            })
        return docs

    def build_cnv_samples(self, cnv_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building CNV_SAMPLES collection ...")
        sample_ids = cnv_df.columns[1:].tolist()
        return [{
            "_key":               sid,
            "sample_id":          sid,
            "cohort":             self.study,
            "data_type":          "cnv_vector",
            "cnv_index_ref":      f"cnv_index_{self.study}",
            "analysis_method":    "ASCAT3",
            "n_genes":            len(cnv_df),
            "values_copy_number": pd.to_numeric(cnv_df[sid], errors="coerce").tolist(),
        } for sid in tqdm(sample_ids, desc="CNV samples")]

    def build_methylation_samples(self, methylation_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building METHYLATION_SAMPLES collection ...")
        sample_ids = methylation_df.columns[1:].tolist()
        return [{
            "_key":                  sid,
            "sample_id":             sid,
            "cohort":                self.study,
            "data_type":             "methylation_vector",
            "methylation_index_ref": f"methylation_index_{self.study}",
            "platform":              "Illumina HumanMethylation27",
            "n_probes":              len(methylation_df),
            "values_beta":           pd.to_numeric(methylation_df[sid], errors="coerce").tolist(),
            "value_range":           "[0.0, 1.0]",
            "description":           "Beta values representing methylation levels at CpG sites.",
        } for sid in tqdm(sample_ids, desc="methylation samples")]

    def build_mirna_samples(self, mirna_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building MIRNA_SAMPLES collection ...")
        sample_ids = mirna_df.columns[1:].tolist()
        return [{
            "_key":              sid,
            "sample_id":         sid,
            "cohort":            self.study,
            "data_type":         "mirna_vector",
            "mirna_index_ref":   f"mirna_index_{self.study}",
            "platform":          "Illumina",
            "n_mirnas":          len(mirna_df),
            "values_expression": pd.to_numeric(mirna_df[sid], errors="coerce").tolist(),
        } for sid in tqdm(sample_ids, desc="miRNA samples")]

    def build_protein_samples(self, protein_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building PROTEIN_SAMPLES collection ...")
        sample_ids = protein_df.columns[1:].tolist()
        return [{
            "_key":               sid,
            "sample_id":          sid,
            "cohort":             self.study,
            "data_type":          "protein_vector",
            "protein_index_ref":  f"protein_index_{self.study}",
            "platform":           "RPPA",
            "n_proteins":         len(protein_df),
            "values_abundance":   pd.to_numeric(protein_df[sid], errors="coerce").tolist(),
        } for sid in tqdm(sample_ids, desc="protein samples")]


# Metadata layer (clinical + survival) -------------------------------------

class MetadataLayerBuilder(CollectionBuilder):

    def build_samples(self, clinical_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building SAMPLES collection ...")
        return [{
            "_key":                 row["sample"],
            "submitter_id":         row["submitter_id"],
            "case_ref":             f"cases/{row['submitter_id']}",
            "project_ref":          f"projects/{row['project_id.project']}",
            "sample_type":          row.get("sample_type.samples"),
            "sample_type_id":       row.get("sample_type_id.samples"),
            "tissue_type":          row.get("tissue_type.samples"),
            "tumor_descriptor":     row.get("tumor_descriptor.samples"),
            "specimen_type":        row.get("specimen_type.samples"),
            "composition":          row.get("composition.samples"),
            "preservation_method":  row.get("preservation_method.samples"),
            "days_to_collection":   row.get("days_to_collection.samples"),
            "entity_type":          "sample",
        } for _, row in tqdm(clinical_df.iterrows(), total=len(clinical_df), desc="samples")]

    def build_cases(self, clinical_df: pd.DataFrame, survival_df: pd.DataFrame) -> List[Dict]:
        logger.info("Building CASES collection ...")
        cases = []
        for patient_id, group in tqdm(clinical_df.groupby("submitter_id"), desc="cases"):
            r = group.iloc[0]
            doc = {
                "_key":          patient_id,
                "project_ref":   f"projects/{r['project_id.project']}",
                "primary_site":  r.get("primary_site"),
                "disease_type":  r.get("disease_type"),
                "demographic": {
                    "gender":         r.get("gender.demographic"),
                    "race":           r.get("race.demographic"),
                    "ethnicity":      r.get("ethnicity.demographic"),
                    "vital_status":   r.get("vital_status.demographic"),
                    "age_at_index":   r.get("age_at_index.demographic"),
                    "days_to_birth":  r.get("days_to_birth.demographic"),
                    "year_of_birth":  r.get("year_of_birth.demographic"),
                    "year_of_death":  r.get("year_of_death.demographic"),
                    "days_to_death":  r.get("days_to_death.demographic"),
                },
                "diagnoses": {
                    "primary_diagnosis":     r.get("primary_diagnosis.diagnoses"),
                    "age_at_diagnosis":      r.get("age_at_diagnosis.diagnoses"),
                    "tumor_grade":           r.get("tumor_grade.diagnoses"),
                    "ajcc_pathologic_stage": r.get("ajcc_pathologic_stage.diagnoses"),
                    "ajcc_pathologic_t":     r.get("ajcc_pathologic_t.diagnoses"),
                    "ajcc_pathologic_n":     r.get("ajcc_pathologic_n.diagnoses"),
                    "ajcc_pathologic_m":     r.get("ajcc_pathologic_m.diagnoses"),
                },
                "entity_type": "case",
            }
            survival_match = survival_df[survival_df["sample"] == r["sample"]]
            if not survival_match.empty:
                doc["survival"] = {
                    "overall_survival_time":   float(survival_match.iloc[0]["OS.time"]) if pd.notna(survival_match.iloc[0]["OS.time"]) else None,
                    "overall_survival_status": int(survival_match.iloc[0]["OS"])        if pd.notna(survival_match.iloc[0]["OS"])      else None,
                }
            cases.append(doc)
        return cases


# ---------------------------------------------------------------------------
# Per-study build pipeline
# ---------------------------------------------------------------------------

def build_for_study(study: str, layers: List[str],
                    mapping_dfs: Dict, lookups: Dict,
                    output_root: Path,
                    do_semantic: bool, do_metadata: bool):
    """Run the build pipeline for a single study, restricted to `layers`."""
    output_dir = output_root / study
    logger.info("=" * 60)
    logger.info(f"Building collections for {study} -> {output_dir}")
    logger.info("=" * 60)

    semantic   = SemanticLayerBuilder(study, mapping_dfs, lookups, output_dir)
    index_b    = QuantitativeIndexBuilder(study, mapping_dfs, lookups, output_dir)
    vector_b   = SampleCentricVectorBuilder(study, mapping_dfs, lookups, output_dir)
    metadata_b = MetadataLayerBuilder(study, mapping_dfs, lookups, output_dir)

    clinical_df = load_omics_data(study, "clinical")

    # Semantic layer ---------------------------------------------------
    if do_semantic:
        semantic.save_collection(semantic.build_gene_nodes(), "genes")
        if clinical_df is not None:
            # Annotate clinical with MONDO mapping when available
            if "mondo" in mapping_dfs:
                mm = mapping_dfs["mondo"]
                if {"_disease_type", "mondo_id"}.issubset(mm.columns):
                    clinical_df = clinical_df.merge(
                        mm[["_disease_type", "mondo_id"]],
                        left_on="disease_type.project", right_on="_disease_type", how="left",
                    ).drop(columns=["_disease_type"])
                    clinical_df.to_csv(output_dir / f"{study}_clinical_with_mondo.csv", index=False)
            semantic.save_collection(semantic.build_project_node(clinical_df), "projects")

    # Metadata layer ---------------------------------------------------
    if do_metadata and clinical_df is not None:
        survival_df = load_omics_data(study, "survival")
        metadata_b.save_collection(metadata_b.build_samples(clinical_df), "samples")
        if survival_df is not None:
            metadata_b.save_collection(metadata_b.build_cases(clinical_df, survival_df), "cases")

    # Quantitative layers ---------------------------------------------
    if "gene_expression" in layers:
        star_tpm = load_omics_data(study, "star_tpm")
        if star_tpm is not None:
            index_b.save_collection(index_b.build_expression_index(star_tpm), "gene_expression_index")
            vector_b.save_collection(vector_b.build_gene_expression_samples(star_tpm), f"gene_expression_samples_{study}")

    if "cnv" in layers:
        cnv_df = load_omics_data(study, "gene-level_ascat3")
        if cnv_df is not None:
            index_b.save_collection(index_b.build_cnv_index(cnv_df), "cnv_index")
            vector_b.save_collection(vector_b.build_cnv_samples(cnv_df), f"cnv_samples_{study}")

    if "mirna" in layers:
        mirna_df = load_omics_data(study, "mirna")
        if mirna_df is not None and "mirna" in mapping_dfs:
            mirna_df = mirna_df.merge(mapping_dfs["mirna"][["miRNA_ID", "hgcn_id"]], on="miRNA_ID", how="left")
        if mirna_df is not None:
            index_b.save_collection(index_b.build_mirna_index(mirna_df), "mirna_index")
            vector_b.save_collection(vector_b.build_mirna_samples(mirna_df), f"mirna_samples_{study}")

    if "protein" in layers:
        protein_df = load_omics_data(study, "protein")
        if protein_df is not None:
            index_b.save_collection(index_b.build_protein_index(protein_df), "protein_index")
            vector_b.save_collection(vector_b.build_protein_samples(protein_df), f"protein_samples_{study}")

    if "methylation" in layers:
        methylation_df = load_omics_data(study, "methylation27")
        if methylation_df is not None:
            index_b.save_collection(index_b.build_methylation_index(methylation_df), "methylation_index")
            vector_b.save_collection(vector_b.build_methylation_samples(methylation_df), f"methylation_samples_{study}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build ArangoDB-ready JSON collections from per-study TCGA/TARGET omic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--studies", nargs="+", metavar="STUDY",
                       help="One or more study IDs (e.g. TCGA-BRCA TCGA-LUAD TARGET-AML).")
    group.add_argument("--cohort", choices=["tcga", "target", "all"],
                       help="Build for an entire cohort. 'all' = TCGA + TARGET.")
    p.add_argument("--layers", nargs="+", choices=QUANT_LAYERS, default=QUANT_LAYERS,
                   help=f"Quantitative layers to build (default: all of {QUANT_LAYERS}).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Root directory for the generated JSON collections.")
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip the semantic layer (genes, projects).")
    p.add_argument("--no-metadata", action="store_true",
                   help="Skip the metadata layer (samples, cases).")
    return p.parse_args()


def resolve_studies(args) -> List[str]:
    if args.studies:
        return args.studies
    if args.cohort == "tcga":   return TCGA_STUDIES
    if args.cohort == "target": return TARGET_STUDIES
    if args.cohort == "all":    return TCGA_STUDIES + TARGET_STUDIES
    return ["TCGA-BRCA"]


def main():
    args = parse_args()
    studies = resolve_studies(args)
    if not args.studies and not args.cohort:
        logger.info("No --studies or --cohort provided; defaulting to TCGA-BRCA.")

    logger.info("=" * 60)
    logger.info(f"build_omics_collections — target: {len(studies)} study(ies)")
    logger.info(f"layers       : {', '.join(args.layers)}")
    logger.info(f"semantic     : {'no' if args.no_semantic else 'yes'}")
    logger.info(f"metadata     : {'no' if args.no_metadata else 'yes'}")
    logger.info(f"output root  : {args.output_root}")
    logger.info("=" * 60)

    mapping_dfs = load_mappings()
    lookups     = create_lookup_dicts(mapping_dfs)

    for study in studies:
        try:
            build_for_study(
                study, args.layers, mapping_dfs, lookups,
                output_root=args.output_root,
                do_semantic=not args.no_semantic,
                do_metadata=not args.no_metadata,
            )
        except Exception as exc:
            logger.exception(f"Build failed for {study}: {exc}")

    logger.info("[OK] build_omics_collections finished.")


if __name__ == "__main__":
    main()
