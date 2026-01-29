import pandas as pd
import polars as pl
from cyvcf2 import VCF

from traitgym.intervals import add_exon, add_tss
from traitgym.variants import (
    COORDINATES,
    NUCLEOTIDES,
    CHROMS,
    filter_snp,
    filter_chroms,
    lift_hg19_to_hg38,
)
