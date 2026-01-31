FINEMAPPING_URL = "https://huggingface.co/datasets/gonzalobenegas/finucane-ukbb-finemapping/resolve/main"
FINEMAPPING_METHODS = ["SuSiE", "FINEMAP"]
COMPLEX_TRAITS = pl.read_csv("config/complex_traits.csv")["trait"].to_list()


rule complex_traits_download_finemapping:
    output:
        "results/complex_traits/finemapping/{trait}/{method}.parquet",
    run:
        url = f"{FINEMAPPING_URL}/UKBB.{wildcards.trait}.{wildcards.method}.tsv.bgz"
        (
            pl.read_csv(
                url,
                separator="\t",
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
            .cast({"chrom": pl.String})
            .pipe(filter_snp)
            .write_parquet(output[0])
        )
