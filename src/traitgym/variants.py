import polars as pl
from liftover import get_lifter

COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def filter_snp(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("ref").is_in(NUCLEOTIDES) & pl.col("alt").is_in(NUCLEOTIDES))


def filter_chroms(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("chrom").is_in(CHROMS))


def lift_hg19_to_hg38(V: pl.DataFrame) -> pl.DataFrame:
    converter = get_lifter("hg19", "hg38")

    def get_new_pos(chrom: str, pos: int) -> int:
        try:
            res = converter[chrom][pos]
            assert len(res) == 1
            new_chrom, new_pos, strand = res[0]
            assert new_chrom.replace("chr", "") == chrom
            return new_pos
        except:
            return -1

    return V.with_columns(
        pl.struct(["chrom", "pos"])
        .map_elements(lambda x: get_new_pos(x["chrom"], x["pos"]), return_dtype=pl.Int64)
        .alias("pos")
    )
