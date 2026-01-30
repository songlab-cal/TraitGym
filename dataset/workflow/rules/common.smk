import pandas as pd
import polars as pl
from cyvcf2 import VCF
from sklearn.metrics import average_precision_score

from traitgym.intervals import add_exon, add_tss
from traitgym.matching import match_features
from traitgym.variants import (
    COORDINATES,
    NUCLEOTIDES,
    CHROMS,
    filter_snp,
    filter_chroms,
    lift_hg19_to_hg38,
)
