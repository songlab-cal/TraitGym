import polars as pl

COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def filter_snp(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("ref").is_in(NUCLEOTIDES) & pl.col("alt").is_in(NUCLEOTIDES))


def filter_chroms(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("chrom").is_in(CHROMS))
