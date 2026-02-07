from gpn.data import Genome
import pandas as pd
import polars as pl
from cyvcf2 import VCF
from sklearn.metrics import average_precision_score

from traitgym.intervals import add_exon, add_tss, build_dataset
from traitgym.matching import match_features
from traitgym.stats import (
    consequence_counts,
    source_consequence_counts,
    consequence_counts_to_tex,
    source_consequence_counts_to_tex,
)
from traitgym.variants import (
    COORDINATES,
    NUCLEOTIDES,
    CHROMS,
    filter_snp,
    filter_chroms,
    lift_hg19_to_hg38,
    check_ref_alt,
)


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget {config[genome_url]} -O {output}"
