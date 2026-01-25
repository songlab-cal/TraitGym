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
        xls = pd.ExcelFile(input[0])
        sheet_names = xls.sheet_names

        dfs = []
        for variant_type in sheet_names[1:]:
            df = pd.read_excel(input[0], sheet_name=variant_type)
            dfs.append(df)
        V = pd.concat(dfs)
        V = V[["Chr", "Position", "Ref", "Alt", "OMIM"]].rename(
            columns={
                "Chr": "chrom",
                "Position": "pos",
                "Ref": "ref",
                "Alt": "alt",
                "OMIM": "trait",
            }
        )
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_chroms(V)
        V = filter_snp(V)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        V = sort_variants(V)
        V.to_parquet(output[0], index=False)
