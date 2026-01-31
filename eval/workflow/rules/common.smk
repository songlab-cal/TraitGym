import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from gpn.data import Genome
from gpn.star.utils import get_llr, normalize_logits
from traitgym.variants import CHROMS, COORDINATES


PRECOMPUTED_MODELS = list(config["precomputed"].keys())
GPN_STAR_MODELS = list(config.get("gpn_star", {}).keys())


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget {config[genome_url]} -O {output}"
