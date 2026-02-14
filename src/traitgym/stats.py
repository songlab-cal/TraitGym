import polars as pl

SOURCE_ORDER = ["omim", "smedley_et_al"]

SOURCE_LABELS = {
    "": "",
    "omim": "OMIM",
    "smedley_et_al": "Smedley et al.",
}


def source_consequence_counts(df: pl.DataFrame) -> pl.DataFrame:
    per_source = (
        df.group_by(["source", "consequence"])
        .agg(pl.count())
        .with_columns(pl.col("source").cast(pl.Enum(SOURCE_ORDER)))
        .sort(["source", "count"], descending=[False, True])
        .with_columns(pl.col("source").cast(pl.String))
    )
    total = pl.DataFrame(
        {"source": [""], "consequence": ["total"], "count": [per_source["count"].sum()]},
        schema_overrides={"count": per_source["count"].dtype},
    )
    return pl.concat([total, per_source])


def consequence_counts(df: pl.DataFrame) -> pl.DataFrame:
    per_consequence = (
        df.get_column("consequence")
        .value_counts()
        .sort("count", descending=True)
    )
    total = pl.DataFrame(
        {"consequence": ["total"], "count": [per_consequence["count"].sum()]},
        schema_overrides={"count": per_consequence["count"].dtype},
    )
    return pl.concat([total, per_consequence])


def source_consequence_counts_to_tex(input_path: str, output_path: str) -> None:
    df = pl.read_parquet(input_path)

    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Source & Consequence & Count \\",
        r"\midrule",
    ]

    for row in df.iter_rows(named=True):
        source = SOURCE_LABELS[row["source"]]
        consequence = row["consequence"]
        if consequence == "total":
            lines.append(f" & Total & {row['count']} " + r"\\")
        else:
            lines.append(f"{source} & \\verb|{consequence}| & {row['count']} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def consequence_counts_to_tex(input_path: str, output_path: str) -> None:
    df = pl.read_parquet(input_path)

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Consequence & Count \\",
        r"\midrule",
    ]

    for row in df.iter_rows(named=True):
        consequence = row["consequence"]
        if consequence == "total":
            lines.append(f"Total & {row['count']} " + r"\\")
        else:
            lines.append(f"\\verb|{consequence}| & {row['count']} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
