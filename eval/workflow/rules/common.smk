import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyBigWig
import seaborn as sns
from sklearn.metrics import average_precision_score

from gpn.data import Genome
from gpn.star.utils import get_llr, normalize_logits
from traitgym.metrics import (
    iid_mean_se,
    mean_reciprocal_rank_scores,
    pairwise_accuracy_scores,
    stratified_bootstrap_se,
)
from traitgym.plotting import get_subset_order_and_aliases, plot_model_bars
from traitgym.variants import CHROMS, COORDINATES


PRECOMPUTED_MODELS = list(config["precomputed"].keys())
PRECOMPUTED_FEATURE_MODELS = PRECOMPUTED_MODELS + ["SpliceAI"]
GPN_STAR_MODELS = list(config.get("gpn_star", {}).keys())


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget {config[genome_url]} -O {output}"


rule abs_llr:
    input:
        "results/features/{dataset}/{model}_LLR.parquet",
    output:
        "results/features/{dataset}/{model}_absLLR.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .with_columns(pl.col("score").abs())
            .write_parquet(output[0])
        )
