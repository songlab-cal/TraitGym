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
                    "beta_posterior",
                    "sd_posterior",
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
            .drop_nulls("pip")
            .write_parquet(output[0])
        )


rule complex_traits_combine_methods:
    input:
        susie="results/complex_traits/finemapping/{trait}/SuSiE.parquet",
        finemap="results/complex_traits/finemapping/{trait}/FINEMAP.parquet",
    output:
        "results/complex_traits/finemapping/{trait}/combined.parquet",
    run:
        (
            pl.read_parquet(input.susie)
            .join(
                pl.read_parquet(input.finemap),
                on=COORDINATES,
                how="inner",
                suffix="_finemap",
            )
            .rename(
                {
                    "pip": "pip_susie",
                    "beta_posterior": "beta_posterior_susie",
                    "sd_posterior": "sd_posterior_susie",
                }
            )
            .with_columns(
                pl.when((pl.col("pip_susie") - pl.col("pip_finemap")).abs() <= 0.05)
                .then((pl.col("pip_susie") + pl.col("pip_finemap")) / 2)
                .otherwise(pl.lit(None))
                .alias("pip"),
            )
            .drop("rsid_finemap")
            .write_parquet(output[0])
        )
