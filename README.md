# CURE: Conditional Unaligned Representation Enhancement for Heterogeneous Software Defect Prediction

This repository contains the implementation of **CURE**, a conditional data augmentation framework for heterogeneous defect prediction (HDP), as described in the paper:

> **CURE: Conditional Unaligned Representation Enhancement for Heterogeneous Software Defect Prediction**  
> *(TSE Revision)*

## Method Overview

CURE improves cross-project defect prediction under heterogeneous feature spaces ($d_s \neq d_t$) through three phases:

1. **Conditional Representation Alignment** — Separate source ($F_s$) and target ($F_t$) encoders map heterogeneous metrics into a shared latent space $Z$, with MMD-based alignment and source classification supervision.
2. **Task-driven Conditional Generation** — A conditional generator $G_\phi$ synthesizes class-consistent pseudo-samples in latent space to alleviate class imbalance and enrich decision boundaries.
3. **Consistency-based Refinement** — KL-divergence consistency regularization stabilizes the classifier on both real and generated latent representations.

### Key Files

| File | Description |
|------|-------------|
| `CURE.py` | Main CURE implementation (Phase I–III) |
| `deep_hdp_baselines.py` | DANN, ADDA, Deep CORAL implementations |
| `hdp_models.py` | Baseline HDP models (CPDP-IFS, CLSUP, FMT, MSMDA) |
| `models/` | Traditional transfer learning models (TCA, JDA, BDA, CORAL, etc.) |
| `run_cure.py` | Run CURE experiments |
| `run_hdp_experiments.py` | Run HDP baselines and CURE-enhanced variants |
| `run_deep_baselines.py` | Run DANN/ADDA/Deep CORAL baselines |
| `run_full_experiment_suite.py` | Full experiment suite (530 transfer pairs) |
| `rq2_cure_diagnostics.py` | Diagnostic metrics (Coverage@ε, KS pass rate) |

## Datasets

The `Datasets/` directory contains four standard SDP benchmarks:

- **AEEEM** — 5 projects, 61 metrics
- **NASA** — 10 projects, 38 metrics  
- **PROMISE** — 10 projects, 20 metrics
- **JIRA** — 4 projects, 22 metrics

Each dataset follows the standard CSV format: rows are software modules/classes, columns are metric features, with a binary `bug` label (1=defective, 0=clean).

## Requirements

```bash
pip install numpy pandas scikit-learn scipy torch
```

## Quick Start

```bash
# Run CURE on a single transfer pair
python run_cure.py --source data/AEEEM/EQ.csv --target data/NASA/CM1.csv --epochs 30

# Run full HDP baseline comparison
python run_hdp_experiments.py --models CPDP_IFS,CLSUP,FMT,MSMDA --dataset AEEEM

# Run deep DA baselines
python run_deep_baselines.py --models DANN,ADDA,DeepCORAL --dataset AEEEM
```

## Baselines Included

- **HDP Baselines**: CPDP-IFS, CLSUP, FMT, MSMDA
- **Deep DA Baselines**: DANN, ADDA, Deep CORAL
- **Traditional Transfer Learning**: TCA, JDA, BDA, CCA+, CORAL
```
## License

MIT License
