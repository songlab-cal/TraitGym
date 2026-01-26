# Regulatory variants associated with Mendelian diseases curatedfrom the literature
# by Smedley et al. (2016). Table S6 from:
# Smedley, Damian, et al. "A whole-genome analysis framework for effective
# identification of pathogenic regulatory variants in Mendelian disease." The
# American Journal of Human Genetics 99.3 (2016): 595-606.


rule smedley_et_al_download:
    output:
        temp("results/smedley_et_al/variants.xslx"),
    shell:
        "wget -O {output} https://ars.els-cdn.com/content/image/1-s2.0-S0002929716302786-mmc2.xlsx"


rule smedley_et_al_process:
    input:
        "results/smedley_et_al/variants.xslx",
    output:
        "results/smedley_et_al/variants.parquet",
    run:
        # Using pandas for Excel reading (polars Excel support is limited)
        xls = pd.ExcelFile(input[0])
        dfs = [pd.read_excel(input[0], sheet_name=name) for name in xls.sheet_names[1:]]
        V = (
            pl.from_pandas(pd.concat(dfs))
            .select(
                pl.col("Chr").str.replace("chr", "").alias("chrom"),
                pl.col("Position").alias("pos"),
                pl.col("Ref").alias("ref"),
                pl.col("Alt").alias("alt"),
                pl.col("OMIM").alias("trait"),
            )
            .pipe(filter_chroms)
            .pipe(filter_snp)
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
        )
        V.write_parquet(output[0])
