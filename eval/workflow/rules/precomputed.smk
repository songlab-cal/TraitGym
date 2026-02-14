rule precomputed_download_cadd:
    output:
        "results/precomputed/CADD.tsv.gz",
        "results/precomputed/CADD.tsv.gz.tbi",
    shell:
        """
        wget {config[precomputed][CADD][url]} -O {output[0]} &&
        wget {config[precomputed][CADD][url]}.tbi -O {output[1]}
        """


rule precomputed_download_gpn_msa:
    output:
        "results/precomputed/GPN-MSA_LLR.tsv.gz",
        "results/precomputed/GPN-MSA_LLR.tsv.gz.tbi",
    shell:
        """
        wget {config[precomputed][GPN-MSA_LLR][url]} -O {output[0]} &&
        wget {config[precomputed][GPN-MSA_LLR][url]}.tbi -O {output[1]}
        """


rule precomputed_extract_chrom:
    input:
        "results/precomputed/{model}.tsv.gz",
    output:
        temp("results/precomputed/{model}/chrom/{chrom}.tsv"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
        model="|".join(PRECOMPUTED_MODELS),
    params:
        region="{chrom}",
    wrapper:
        "v7.3.0/bio/tabix/query"


rule precomputed_to_parquet_gpn_msa:
    input:
        "results/precomputed/GPN-MSA_LLR/chrom/{chrom}.tsv",
    output:
        "results/precomputed/GPN-MSA_LLR/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        pl.read_csv(
            input[0],
            has_header=False,
            separator="\t",
            new_columns=COORDINATES + ["score"],
            schema_overrides={"chrom": str, "score": pl.Float32},
        ).write_parquet(output[0])


rule precomputed_to_parquet_cadd:
    input:
        "results/precomputed/CADD/chrom/{chrom}.tsv",
    output:
        "results/precomputed/CADD/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        pl.read_csv(
            input[0],
            has_header=False,
            separator="\t",
            columns=[0, 1, 2, 3, 4],
            new_columns=COORDINATES + ["score"],
            schema_overrides={"chrom": str, "score": pl.Float32},
        ).write_parquet(output[0])


rule precomputed_features:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        parquets=expand(
            "results/precomputed/{{model}}/chrom/{chrom}.parquet", chrom=CHROMS
        ),
    output:
        "results/features/{dataset}/{model}.parquet",
    wildcard_constraints:
        model="|".join(PRECOMPUTED_FEATURE_MODELS),
    threads: workflow.cores
    run:
        V = pl.read_parquet(input.dataset, columns=COORDINATES)
        results = []
        for path, chrom in zip(input.parquets, CHROMS):
            chrom_variants = V.filter(pl.col("chrom") == chrom).lazy()
            precomputed_lf = pl.scan_parquet(path)
            joined = chrom_variants.join(
                precomputed_lf,
                on=COORDINATES,
                how="left",
                maintain_order="left",
            ).collect(engine="streaming")
            results.append(joined)
        pl.concat(results).select("score").write_parquet(output[0])
