import polars as pl
import polars_bio as pb

from traitgym.variants import CHROMS, NON_EXONIC


GTF_COLUMNS = [
    "chrom",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


def load_annotation(path: str) -> pl.DataFrame:
    """Load a GTF annotation file into a polars DataFrame.

    Converts from 1-based GTF coordinates to 0-based BED-like coordinates.

    Args:
        path: Path to the GTF file (can be gzipped).

    Returns:
        DataFrame with columns [chrom, source, feature, start, end, score,
        strand, frame, attribute]. Start is 0-based.
    """
    return (
        pl.read_csv(
            path,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            new_columns=GTF_COLUMNS,
            schema_overrides={"chrom": pl.String},
        )
        .with_columns(pl.col("start") - 1)  # GTF to BED conversion
        .filter(pl.col("strand").is_in(["+", "-"]))
    )


def get_tss(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract transcription start sites from an annotation DataFrame.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns [chrom, start, end, gene_id] containing
        TSS positions (1bp intervals) from protein-coding transcripts.
    """
    return (
        ann.filter(pl.col("feature") == "transcript")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'gene_id "([^;]*)";')
            .alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(pl.col("transcript_biotype") == "protein_coding")
        .with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.col("end") - 1)
            .alias("tss_start"),
        )
        .with_columns((pl.col("tss_start") + 1).alias("tss_end"))
        .select(
            pl.col("chrom"),
            pl.col("tss_start").alias("start"),
            pl.col("tss_end").alias("end"),
            pl.col("gene_id"),
        )
        .unique()
        .sort(["chrom", "start", "end"])
    )


def get_exon(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract protein-coding exons from an annotation DataFrame.

    Args:
        ann: Annotation DataFrame from load_annotation().

    Returns:
        DataFrame with columns [chrom, start, end, gene_id] containing
        deduplicated exons from protein-coding transcripts, filtered to
        canonical chromosomes and sorted by position.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'gene_id "([^;]*)";')
            .alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(
            pl.col("transcript_biotype") == "protein_coding",
            pl.col("chrom").is_in(CHROMS),
        )
        .select(["chrom", "start", "end", "gene_id"])
        .unique()
        .sort(["chrom", "start", "end"])
    )


def add_exon(
    V: pl.DataFrame,
    exon: pl.DataFrame,
    exon_proximal_dist: int = 100,
) -> pl.DataFrame:
    """Add exon distance and consequence_final column.

    Args:
        V: Variant DataFrame with columns [chrom, pos, consequence,
            consequence_cre, ...].
        exon: Exon intervals from get_exon().
        exon_proximal_dist: Distance threshold for exon_proximal consequence.

    Returns:
        DataFrame with added columns: exon_dist, exon_closest_gene_id,
        consequence_final.
    """
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    result = pb.nearest(
        V_intervals,
        exon,
        cols1=("chrom", "start", "end"),
        cols2=("chrom", "start", "end"),
        suffixes=("", "_exon"),
        output_type="polars.DataFrame",
    )
    return result.select(
        pl.exclude("start", "end", "chrom_exon", "start_exon", "end_exon", "distance", "gene_id_exon"),
        pl.col("distance").alias("exon_dist"),
        pl.col("gene_id_exon").alias("exon_closest_gene_id"),
        pl.when(
            pl.col("consequence") == "intron_variant",
            pl.col("distance") <= exon_proximal_dist,
        )
        .then(pl.lit("exon_proximal"))
        .otherwise(pl.col("consequence_cre"))
        .alias("consequence_final"),
    )


def add_tss(
    V: pl.DataFrame,
    tss: pl.DataFrame,
    tss_proximal_dist: int = 1000,
) -> pl.DataFrame:
    """Add TSS distance and update consequence_final column.

    Args:
        V: Variant DataFrame with columns [chrom, pos, consequence,
            consequence_final, ...].
        tss: TSS intervals from get_tss().
        tss_proximal_dist: Distance threshold for tss_proximal consequence.

    Returns:
        DataFrame with added columns: tss_dist, tss_closest_gene_id.
        Updates consequence_final to tss_proximal where applicable.
    """
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    result = pb.nearest(
        V_intervals,
        tss,
        cols1=("chrom", "start", "end"),
        cols2=("chrom", "start", "end"),
        suffixes=("", "_tss"),
        output_type="polars.DataFrame",
    )
    return result.select(
        pl.exclude(
            "start", "end", "chrom_tss", "start_tss", "end_tss",
            "distance", "gene_id_tss", "consequence_final",
        ),
        pl.col("distance").alias("tss_dist"),
        pl.col("gene_id_tss").alias("tss_closest_gene_id"),
        pl.when(
            pl.col("consequence").is_in(NON_EXONIC),
            pl.col("distance") <= tss_proximal_dist,
        )
        .then(pl.lit("tss_proximal"))
        .otherwise(pl.col("consequence_final"))
        .alias("consequence_final"),
    )
