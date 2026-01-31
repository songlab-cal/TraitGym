FINEMAPPING_URL = "https://huggingface.co/datasets/gonzalobenegas/finucane-ukbb-finemapping/resolve/main"
FINEMAPPING_METHODS = ["SuSiE", "FINEMAP"]
COMPLEX_TRAITS = pl.read_csv("config/complex_traits.csv")["trait"].to_list()


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
                    pl.when((pl.col("pip_susie") - pl.col("pip_finemap")).abs() <= 0.05)
                    .then((pl.col("pip_susie") + pl.col("pip_finemap")) / 2)
                    .otherwise(pl.lit(None))
                )
                .otherwise(pl.coalesce("pip_susie", "pip_finemap"))
                .alias("pip"),
            )
            .select([*COORDINATES, "rsid", "pip"])
            .write_parquet(output[0])
        )
