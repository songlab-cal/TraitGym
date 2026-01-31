FINEMAPPING_URL = "https://huggingface.co/datasets/gonzalobenegas/finucane-ukbb-finemapping/resolve/main"
FINEMAPPING_METHODS = ["SuSiE", "FINEMAP"]
FINEMAPPING_PIP_DIFF_THRESHOLD = 0.05
HIGH_PIP_THRESHOLD = 0.9
LOW_PIP_THRESHOLD = 0.01
COMPLEX_TRAITS = pl.read_csv("config/complex_traits.csv")["trait"].to_list()


rule complex_traits_download_all_finemapping:
    input:
        expand(
            "results/complex_traits/finemapping/{trait}/{method}.parquet",
            trait=COMPLEX_TRAITS,
            method=FINEMAPPING_METHODS,
        ),


rule complex_traits_download_finemapping:
    output:
        "results/complex_traits/finemapping/{trait}/{method}.parquet",
    wildcard_constraints:
        method="|".join(FINEMAPPING_METHODS),
    run:
        url = f"{FINEMAPPING_URL}/UKBB.{wildcards.trait}.{wildcards.method}.tsv.bgz"
        (
            pl.read_csv(
                url,
                separator="\t",
                null_values=["NA"],
                schema_overrides={"chromosome": pl.String},
                columns=[
                    "chromosome",
                    "position",
                    "allele1",
                    "allele2",
                    "rsid",
                    "pip",
                ],
            )
            .rename(
                {
                    "chromosome": "chrom",
                    "position": "pos",
                    "allele1": "ref",
                    "allele2": "alt",
                }
            )
            .pipe(filter_snp)
            .write_parquet(output[0])
        )


rule complex_traits_combine_methods:
    input:
        susie="results/complex_traits/finemapping/{trait}/SuSiE.parquet",
        finemap="results/complex_traits/finemapping/{trait}/FINEMAP.parquet",
    output:
        "results/complex_traits/finemapping/{trait}/combined.parquet",
    run:
        pip_defined_in_both_methods = (
            pl.col("pip_susie").is_not_null() & pl.col("pip_finemap").is_not_null()
        )
        (
            pl.read_parquet(input.susie)
            .join(
                pl.read_parquet(input.finemap),
                on=COORDINATES,
                how="full",
                suffix="_finemap",
            )
            .rename({"pip": "pip_susie", "rsid": "rsid_susie"})
            .with_columns(
                *(pl.coalesce(col, f"{col}_finemap").alias(col) for col in COORDINATES),
                pl.coalesce("rsid_susie", "rsid_finemap").alias("rsid"),
                pl.when(pip_defined_in_both_methods)
                .then(
                    pl.when(
                        (pl.col("pip_susie") - pl.col("pip_finemap")).abs()
                        <= FINEMAPPING_PIP_DIFF_THRESHOLD
                    )
                    .then((pl.col("pip_susie") + pl.col("pip_finemap")) / 2)
                    .otherwise(pl.lit(None))
                )
                .otherwise(pl.coalesce("pip_susie", "pip_finemap"))
                .alias("pip"),
            )
            .select([*COORDINATES, "rsid", "pip"])
            .write_parquet(output[0])
        )


rule complex_traits_aggregate_traits:
    input:
        expand(
            "results/complex_traits/finemapping/{trait}/combined.parquet",
            trait=COMPLEX_TRAITS,
        ),
    output:
        "results/complex_traits/finemapping/aggregated.parquet",
    run:
        any_null_pip = pl.col("pip").is_null().any()
        (
            pl.concat(
                [
                    pl.read_parquet(path).with_columns(trait=pl.lit(trait))
                    for path, trait in zip(input, COMPLEX_TRAITS)
                ]
            )
            .with_columns(
                pl.when(pl.col("pip") > HIGH_PIP_THRESHOLD)
                .then(pl.col("trait"))
                .otherwise(pl.lit(None))
                .alias("trait")
            )
            .group_by(COORDINATES)
            .agg(
                pl.col("rsid").first(),
                pl.col("pip").max(),
                any_null_pip.alias("any_null_pip"),
                pl.col("trait").drop_nulls().unique(),
            )
            .with_columns(pl.col("trait").list.sort().list.join(",").alias("traits"))
            .with_columns(
                pl.when(pl.col("pip") > HIGH_PIP_THRESHOLD)
                .then(pl.lit(True))
                .when((pl.col("pip") < LOW_PIP_THRESHOLD) & ~pl.col("any_null_pip"))
                .then(pl.lit(False))
                .otherwise(pl.lit(None))
                .alias("label")
            )
            .filter(pl.col("label").is_not_null())
            .drop(["trait", "any_null_pip"])
            .sort(COORDINATES)
            .write_parquet(output[0])
        )
