# OmicsCouncil

**Deliberation as Representation: Interpretable Multi-Source Biomedical Embeddings
from Knowledge-Grounded Agent Reasoning.**

Reference implementation for the REHMED 2026 (ECML-PKDD) workshop submission by
the DIETI / IKNOS Lab, University of Naples Federico II.

OmicsCouncil represents a sample not by an opaque latent vector but by the
**trace of a multi-agent deliberation**: modality-specific experts emit typed
evidence claims, cross-examine each other, and a knowledge-graph arbiter grades
and prunes the surviving claims. The verified trace is encoded into a fixed,
fully interpretable vector that a lightweight classifier consumes.

The whole experimental suite is **CPU-only and runs in under a minute** — the
demonstration dataset (Breast Cancer Wisconsin) ships inside scikit-learn, and
the default agents are deterministic statistical reasoners. The architecture is
**domain-agnostic**: a new domain is a new YAML config (and, if needed, a new
data loader), with no change to the deliberation logic. An optional `LLMAgent`
hook turns the same protocol into the full LLM-driven instantiation.

## Quick start

```bash
bash scripts/setup.sh           # create .venv and install deps (seconds)
source .venv/bin/activate
bash scripts/run_quick.sh       # smoke test (one classification run)
bash scripts/run_all.sh         # full pipeline: classification + ablation + robustness
```

Run a single experiment on any config:

```bash
python experiments/run_classification.py --config configs/breast_cancer.yaml
python experiments/run_ablation.py       --config configs/breast_cancer.yaml
python experiments/run_robustness.py     --config configs/breast_cancer.yaml
```

Results are printed as tables and saved as JSON under `results/`.

## Layout

```
REHMED/
├── configs/breast_cancer.yaml        # domain configuration (the only domain-specific file)
├── src/omicscouncil/                 # framework package
│   ├── config.py                     # typed config loader
│   ├── data.py                       # loaders + modality slicing (extensible)
│   ├── claims.py                     # Claim / Reaction / DeliberationTrace
│   ├── agents.py                     # Agent ABC, StatisticalAgent, LLMAgent stub
│   ├── knowledge.py                  # KnowledgePrior + graph-mediated arbiter
│   ├── council.py                    # the 3-round deliberation protocol
│   ├── representation.py             # trace -> Z-sym vector (interpretable schema)
│   ├── pipeline.py                   # end-to-end fit / transform / predict
│   └── metrics.py                    # classification + trustworthy-rep metrics
├── experiments/                      # runnable studies
├── scripts/                          # setup + pipeline bash scripts
├── docs/CODE_EXPLAINED.md            # detailed walkthrough of the code
└── paper/main.tex                    # the LNCS paper
```

## Adapting to a new domain

1. Add a loader in `src/omicscouncil/data.py` (decorate with `@register_loader`)
   returning `(X, feature_names, y, class_names)`; or reuse an existing one.
2. Write a config: declare `modalities` (by feature-name filters), the
   `relations` vocabulary, the `knowledge` source (derive from data or point at
   a real KG edge list), and `reference_markers` for the interpretability metric.
3. Run the experiment scripts with `--config your_domain.yaml`.

See `docs/CODE_EXPLAINED.md` for the full rationale and component-by-component
explanation.
