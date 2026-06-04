# OmicsCouncil

**Deliberation as Representation: Knowledge-Grounded Multi-Agent Embeddings for Multi-Omic Breast-Cancer Subtyping.**

OmicsCouncil represents a patient not by an opaque latent vector but by the
**trace of a multi-agent deliberation**: modality-specific experts emit typed
evidence claims, cross-examine each other, and a knowledge-graph arbiter grades
and prunes the surviving claims. The verified trace is encoded into a fixed,
fully interpretable vector that a lightweight classifier consumes.

The architecture is **domain-agnostic**: a new domain is a new YAML config (and,
if needed, a new data loader), with no change to the deliberation logic. An
optional `LLMAgent` backend (Groq + Qwen3) turns the same protocol into an
LLM-driven instantiation; we ship it but, as discussed in the paper, it
underperforms the deterministic statistical agent on this task.

---

## Two instantiations

The repo ships with two ready-to-run configurations that exercise the same
framework at very different scales:

| | **Lightweight PoC** | **Medium-weight (paper)** |
|---|---|---|
| Config | `configs/breast_cancer.yaml` | `configs/tcga_brca.yaml` |
| Dataset | Breast Cancer Wisconsin (ships with sklearn) | TCGA-BRCA, 4 omic layers |
| Samples | 569 | 272 (after 4-way intersection + PAM50) |
| Modalities | 3 statistical views of FNA cytology | RNA / methylation / miRNA / CNV |
| Knowledge prior | data-derived feature→class graph | PheKnowLator subset (1-hop, 362k triples) |
| Arbiter mode | `weighted_lookup` (exact-edge) | `pkt_anchored` (Option C: subject grounding) |
| Runtime | seconds, CPU | minutes, CPU |
| External downloads | none | PheKnowLator nodes/edges + PAM50 labels |

The lightweight path is the framework's smoke test (and the entry point for new
contributors). The medium-weight path is what the paper reports.

---

## Quick start

```bash
# 1. Environment (Python 3.10+)
bash scripts/setup.sh             # POSIX (Linux / macOS / WSL / Git-Bash)
source .venv/bin/activate
```

On Windows / PowerShell, `scripts/setup.sh` will not run; create the venv
manually instead:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Lightweight PoC (Wisconsin, < 1 min)

```bash
python experiments/run_classification.py --config configs/breast_cancer.yaml
python experiments/run_ablation.py       --config configs/breast_cancer.yaml
python experiments/run_robustness.py     --config configs/breast_cancer.yaml
```

### Medium-weight (TCGA-BRCA, paper-scale)

Four artefacts must exist before the medium-weight run.

```bash
# (a) The four TCGA-BRCA omic matrices (Xena format) from the UCSC GDC Hub.
#     Default selection (BRCA, 7 data types incl. star_tpm/methylation27/mirna/
#     gene-level_ascat3) lands under data/omics/TCGA-BRCA/:
python src/data/download_omics.py
#     Sources / catalogue: data/docs/Layer_Structure_Semantic.md.

# (b) PAM50 labels. Download the cBioPortal clinical TSV from
#     https://www.cbioportal.org/study/clinicalData?id=brca_tcga_pan_can_atlas_2018
#     and place it at:
#       data/omics/TCGA-BRCA/brca_tcga_pan_can_atlas_2018_clinical_data.tsv
#     Then extract the per-sample PAM50 calls:
python src/data/download_pam50.py
#     -> data/omics/TCGA-BRCA/pam50_labels.tsv (981 labelled samples)

# (c) PheKnowLator property graph. Two stages: download the v3.0.2 Zenodo build,
#     then convert the RDF dump into nodes.json / edges.zip:
python src/data/download_pkt.py
#     -> data/pkt/builds/v3.0.2/{PKT.nt.tar.gz, PKT_NodeLabels_with_metadata_v3.0.2.csv}
python src/data/build_property_graph.py
#     -> data/kg/PKT/{nodes.json (~500 MB), edges.zip (~225 MB)}
#
# (d) PheKnowLator subset induced on the selected feature space.
#     Build with --top-k matching `top_features_per_modality` in the config:
python src/data/build_pkt_brca_subset.py --top-k 3000 --n-hops 1
#     -> data/kg/PKT/brca_subset.json (~362k triples)

# Single 70/30 split with all four models + trustworthy metrics:
python experiments/run_classification.py --config configs/tcga_brca.yaml

# 5-fold stratified CV (the numbers reported in the paper):
python experiments/run_classification_cv.py --config configs/tcga_brca.yaml --folds 5

# Composite results figure (4 panels) for the paper:
python src/plots.py
#     -> paper/figures/results_composite.{pdf,png}
```

### Optional: LLMAgent (Groq + Qwen3)

```bash
# Provide a Groq API key in src/api_keys.json:  {"groq": "gsk_...", ...}
# Then switch the agent type:
python experiments/run_classification.py --config configs/tcga_brca_llm.yaml
```

LLM responses are cached on disk under `results/.llm_cache/` (hash-keyed by
model + prompt) so re-runs are free. A full TCGA-BRCA pass takes ~30–50 min on
the Groq free tier on first execution; instant on subsequent runs.

---

## Repository layout

```
OmicsCouncil/
├── configs/
│   ├── breast_cancer.yaml          # lightweight PoC config (Wisconsin)
│   ├── tcga_brca.yaml              # medium-weight default (statistical agents)
│   └── tcga_brca_llm.yaml          # LLM-agent variant of the above
│
├── src/
│   ├── omicscouncil/               # framework package (domain-agnostic)
│   │   ├── config.py               # typed YAML config loader
│   │   ├── data.py                 # MultiModalDataset + dual loader registry
│   │   ├── loaders_tcga.py         # TCGA-BRCA loader (HGNC canonicalisation, MAD/ANOVA)
│   │   ├── claims.py               # Claim / Reaction / DeliberationTrace
│   │   ├── agents.py               # StatisticalAgent + LLMAgent (Qwen3, disk cache)
│   │   ├── knowledge.py            # KnowledgePrior + dual arbiter modes
│   │   ├── council.py              # 3-round deliberation protocol
│   │   ├── representation.py       # trace -> Z-sym vector (fixed schema)
│   │   ├── pipeline.py             # end-to-end fit / transform / predict
│   │   └── metrics.py              # classification + trustworthy-rep metrics
│   ├── data/                       # data utilities
│   │   ├── download_omics.py       # fetch TCGA/TARGET matrices from UCSC Xena GDC Hub
│   │   ├── download_pam50.py       # extract PAM50 calls from cBioPortal TSV
│   │   ├── download_pkt.py         # fetch PheKnowLator v3.0.2 build from Zenodo
│   │   ├── build_property_graph.py # PKT RDF dump -> nodes.json + edges.json
│   │   ├── build_pkt_brca_subset.py# stream PheKnowLator -> brca_subset.json
│   │   ├── omics_utils.py          # TCGA matrix readers + path config
│   │   └── pkt_utils.py            # PheKnowLator helpers
│   ├── llm_lite_async.py           # Groq + OpenRouter client (sync + async)
│   └── plots.py                    # composite results figure for the paper
│
├── experiments/                    # runnable studies
│   ├── _common.py                  # shared loaders + 3 baselines (LogReg, MLP, late-fusion)
│   ├── run_classification.py       # single split, OC + baselines + trust metrics
│   ├── run_classification_cv.py    # 5-fold stratified CV (paper protocol)
│   ├── run_ablation.py             # no-cross-exam / no-arbiter / untyped
│   └── run_robustness.py           # missing-modality sweep
│
├── scripts/                        # bash launchers (Wisconsin only for now)
│   ├── setup.sh                    # venv + pip install
│   ├── run_quick.sh                # 1-shot smoke test
│   └── run_all.sh                  # classification + ablation + robustness
│
├── data/
│   ├── omics/TCGA-BRCA/            # Xena .tsv.gz matrices + PAM50 labels
│   ├── pkt/builds/v3.0.2/          # raw PKT Zenodo dump + property_graph/ output
│   ├── kg/PKT/                     # nodes.json + edges.zip (from build_property_graph)
│   │                               #   + brca_subset.json (from build_pkt_brca_subset)
│   ├── mappings/                   # biomart, HM27 probemap, miRNA<->HGNC (shipped zips)
│   └── docs/                       # KG schema notes
│
├── docs/                           # design notes and rationale
│   ├── redirection_report_optC.md  # canonical document for the medium-weight redesign
│   ├── TODO_omicscouncil_medium_weight.md  # status + remaining polish
│   └── CODE_EXPLAINED.md           # framework walkthrough
│
├── paper/
│   ├── main.tex                    # 12-page LNCS submission
│   ├── main_v0.tex                 # archived lightweight-only draft
│   ├── references.bib
│   └── figures/results_composite.{pdf,png}
│
├── results/                        # JSON dumps from every experiment script
└── requirements.txt
```

---

## Adapting to a new domain

There are two extension points, in order of effort:

1. **Same flat-matrix dataset, new modality split.** Add a `@register_loader`
   in `src/omicscouncil/data.py` returning `(X, feature_names, y, class_names)`;
   write a YAML config that slices it via `feature_filter` tokens. This is the
   Wisconsin path.

2. **True multi-modal dataset (separate matrices per omic layer).** Add a
   `@register_dataset_loader` returning a fully assembled
   `MultiModalDataset` (per-modality matrices + label vector). The TCGA-BRCA
   loader in `src/omicscouncil/loaders_tcga.py` is the template.

In both cases, no change to the deliberation logic, the arbiter, or the Z-sym
encoder is required.

---

## Results summary (5-fold CV, TCGA-BRCA PAM50, N=272)

| Model | Accuracy | Macro-F1 | Bal.acc | ECE ↓ |
|---|---:|---:|---:|---:|
| Concat+LogReg | **0.835 ± 0.051** | **0.821 ± 0.046** | **0.825 ± 0.051** | 0.114 ± 0.053 |
| Concat+MLP | 0.809 ± 0.048 | 0.782 ± 0.040 | 0.781 ± 0.054 | 0.180 ± 0.043 |
| Late-fusion | 0.824 ± 0.071 | 0.798 ± 0.076 | 0.778 ± 0.079 | 0.128 ± 0.024 |
| **OmicsCouncil** | 0.791 ± 0.055 | 0.760 ± 0.042 | 0.749 ± 0.041 | **0.091 ± 0.024** |

OmicsCouncil trades ~4 accuracy points for the best calibration of the four
models (20–50% lower ECE), the lowest fold-to-fold variance, and an
interpretable representation: containment rate 0.40, marker recovery ≥1 =
1.00, marker recovery ≥3 = 0.97 (mean 3.7 canonical PAM50 markers per
verified trace), faithfulness 0.083. See `docs/redirection_report_optC.md`
for the full rationale and `paper/main.tex` for the discussion.

---

## Citation

Pending acceptance at REHMED 2026 (ECML-PKDD).
