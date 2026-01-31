import polars as pl
from liftover import get_lifter

COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")
CHROMS = sorted([str(i) for i in range(1, 23)] + ["X", "Y"])
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
NON_EXONIC = [
    "intergenic_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
]


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT[base] for base in reversed(seq))


def filter_snp(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("ref").is_in(NUCLEOTIDES) & pl.col("alt").is_in(NUCLEOTIDES))


def filter_chroms(V: pl.DataFrame) -> pl.DataFrame:
    return V.filter(pl.col("chrom").is_in(CHROMS))


def lift_hg19_to_hg38(V: pl.DataFrame) -> pl.DataFrame:
    converter = get_lifter("hg19", "hg38", one_based=True)
    original_columns = V.columns

    def get_new_coords(row: dict) -> dict:
        try:
            res = converter[row["chrom"]][row["pos"]]
            assert len(res) == 1
            chrom, pos, strand = res[0]
            chrom = chrom.replace("chr", "")
            ref = row["ref"]
            alt = row["alt"]
            if strand == "-":
                ref = reverse_complement(ref)
                alt = reverse_complement(alt)
            return {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}
        except:
            return {"chrom": row["chrom"], "pos": -1, "ref": row["ref"], "alt": row["alt"]}

    return (
        V.with_columns(
            pl.struct(["chrom", "pos", "ref", "alt"])
            .map_elements(
                get_new_coords,
                return_dtype=pl.Struct({"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8}),
            )
            .alias("_coords")
        )
        .drop("chrom", "pos", "ref", "alt")
        .unnest("_coords")
        .select(original_columns)
    )


def check_ref_alt(V: pl.DataFrame, genome) -> pl.DataFrame:
    """
    Check and fix ref/alt alleles against the reference genome.

    For each variant:
    1. Get the reference nucleotide from the genome at that position
    2. If ref doesn't match, swap ref and alt
    3. Filter out variants where neither ref nor alt matches the reference

    Args:
        V: DataFrame with chrom, pos, ref, alt columns
        genome: Genome object with get_nuc(chrom, pos) method

    Returns:
        DataFrame with corrected ref/alt and filtered to valid variants
    """

    def get_ref_nuc(row: dict) -> str:
        return genome.get_nuc(row["chrom"], row["pos"]).upper()

    original_columns = V.columns

    return (
        V.with_columns(
            pl.struct(["chrom", "pos"])
            .map_elements(get_ref_nuc, return_dtype=pl.Utf8)
            .alias("_ref_nuc")
        )
        .with_columns(_needs_swap=(pl.col("ref") != pl.col("_ref_nuc")))
        .with_columns(
            ref=pl.when(pl.col("_needs_swap")).then(pl.col("alt")).otherwise(pl.col("ref")),
            alt=pl.when(pl.col("_needs_swap")).then(pl.col("ref")).otherwise(pl.col("alt")),
        )
        .filter(pl.col("ref") == pl.col("_ref_nuc"))
        .select(original_columns)
    )
