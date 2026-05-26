"""Extract PAM50 calls for TCGA-BRCA from the cBioPortal clinical export.

Input (manually downloaded from
https://www.cbioportal.org/study/clinicalData?id=brca_tcga_pan_can_atlas_2018):

    data/omics/TCGA-BRCA/brca_tcga_pan_can_atlas_2018_clinical_data.tsv

Relevant columns:
  - ``Sample ID``  e.g. TCGA-3C-AAAU-01  (matches our omic matrices once the
    trailing vial letter, e.g. '-01A', is normalized to '-01').
  - ``Subtype``    e.g. BRCA_LumA / BRCA_LumB / BRCA_Basal / BRCA_Her2 /
    BRCA_Normal / NA. We strip the 'BRCA_' prefix.

Output: ``data/omics/TCGA-BRCA/pam50_labels.tsv`` with columns
``sample_id`` (TCGA-XX-YYYY-01) and ``pam50`` (LumA/LumB/Basal/Her2/Normal).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "omics" / "TCGA-BRCA" / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"
OUT = ROOT / "data" / "omics" / "TCGA-BRCA" / "pam50_labels.tsv"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(
            f"Missing {SRC}.\nDownload it from "
            "https://www.cbioportal.org/study/clinicalData?id=brca_tcga_pan_can_atlas_2018"
        )
    df = pd.read_csv(SRC, sep="\t", low_memory=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    sample_col = "Sample ID"
    subtype_col = "Subtype"
    if subtype_col not in df.columns or sample_col not in df.columns:
        raise KeyError(f"Expected columns '{sample_col}' and '{subtype_col}' in {SRC}")

    out = df[[sample_col, subtype_col]].copy()
    out.columns = ["sample_id", "pam50"]

    out["pam50"] = (
        out["pam50"].astype(str).str.strip().str.replace("^BRCA_", "", regex=True)
    )
    out = out[out["pam50"].isin(["LumA", "LumB", "Basal", "Her2", "Normal"])]
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["sample_id"])

    print(f"PAM50-labelled samples: {len(out):,}")
    print(out["pam50"].value_counts().to_string())
    out.to_csv(OUT, sep="\t", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
