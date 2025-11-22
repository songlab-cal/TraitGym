# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TraitGym is a benchmarking framework for evaluating DNA sequence models on causal regulatory variant prediction tasks in human genetics. It compares genomic language models (gLMs) like GPN-MSA, Enformer, Borzoi, and others on mendelian and complex trait variant datasets.

## Installation & Setup

```bash
# Install dependencies
uv sync

# For development (includes pre-commit hooks)
uv sync --group dev
uv run pre-commit install
```

## Running Snakemake Workflows

The project uses Snakemake for pipeline orchestration. All rules are defined in `workflow/Snakefile` and `workflow/rules/`.

```bash
# Run a specific target
uv run snakemake --cores all <path>

# Dry run to see execution plan
uv run snakemake --cores all <path> --dry-run

# Touch existing files to prevent re-running
uv run snakemake --cores all <path> --touch
```

Example targets:
```bash
# Zero-shot LLR evaluation
results/dataset/complex_traits_matched_9/AUPRC_by_chrom_weighted_average/all/GPN-MSA_absLLR.plus.score.csv

# Logistic regression/linear probing
results/dataset/complex_traits_matched_9/AUPRC_by_chrom_weighted_average/all/GPN-MSA.LogisticRegression.chrom.csv
```

For Apptainer/container support:
```bash
uv run snakemake --sdm apptainer --apptainer-args "--bind /scratch/users/gbenegas"
```

## Code Formatting

```bash
# Run all pre-commit hooks
uv run pre-commit run
```

Uses `snakefmt` for formatting both `.smk` and `.py` files.

## Architecture

### Snakemake Pipeline Structure

- `workflow/Snakefile`: Main entry point, includes all rule files
- `workflow/rules/common.smk`: Shared utilities, constants, and common rules (variant filtering, matching, VEP annotation)
- `workflow/rules/data/*.smk`: Data preparation rules (ClinVar, gnomAD, GWAS, eQTL, etc.)
- `workflow/rules/features/*.smk`: Feature extraction for each model (GPN-MSA, Enformer, Borzoi, CADD, etc.)
- `workflow/rules/model.smk`: Classifier training and evaluation rules
- `config/config.yaml`: Configuration for datasets, model paths, feature sets, and hyperparameters

### Key Data Flow

1. **Data preparation**: Raw variant data → filtered variants with annotations → matched case/control datasets
2. **Feature extraction**: Variants → model-specific features (LLR, embeddings, inner products)
3. **Evaluation**: Features → predictions via zero-shot or trained classifiers → metrics (AUPRC, AUROC)

### Result Paths

```
results/dataset/{dataset}/test.parquet           # Main test set
results/dataset/{dataset}/subset/{subset}.parquet # Variant subsets
results/dataset/{dataset}/features/{model}.parquet # Model features
results/dataset/{dataset}/preds/{subset}/{model}.parquet # Predictions
results/dataset/{dataset}/{metric}/{subset}/{model}.csv # Metrics
```

### Feature Sets

Feature combinations for ensemble models are defined in `config/config.yaml` under `feature_sets`. To evaluate a new model:
1. Place features in `results/dataset/{dataset}/features/{features}.parquet`
2. For zero-shot: `snakemake results/dataset/{dataset}/{metric}/all/{features}.{sign}.{feature}.csv`
3. For logistic regression: Define feature set in config, then run classifier target

### Key Constants (workflow/rules/common.smk)

- `COORDINATES = ["chrom", "pos", "ref", "alt"]`: Standard variant coordinate columns
- `CHROMS`: Chromosomes 1-22 (autosomes only)
- `NUCLEOTIDES = list("ACGT")`: Valid nucleotides for SNP filtering

### Datasets

- `mendelian_traits_matched_9`: Mendelian disease variants (ClinVar pathogenic) matched 1:9 with controls
- `complex_traits_matched_9`: Complex trait fine-mapped variants matched 1:9 with controls
