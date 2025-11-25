rule precomputed_download_cadd:
    output:
        "results/precomputed/CADD.tsv.gz",
        "results/precomputed/CADD.tsv.gz.tbi",
    shell:
        """
        wget https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz -O {output[0]} &&
        wget https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz.tbi -O {output[1]}
        """


rule precomputed_download_gpn_msa:
    output:
        "results/precomputed/GPN-MSA_LLR.tsv.gz",
        "results/precomputed/GPN-MSA_LLR.tsv.gz.tbi",
    shell:
        """
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz -O {output[0]} &&
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz.tbi -O {output[1]}
        """


# requires calling snakemake with --use-conda
rule precomputed_extract_chrom:
    input:
        "results/precomputed/{model}.tsv.gz",
    output:
        temp("results/precomputed/{model}/chrom/{chrom}.tsv"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
        model="CADD|GPN-MSA_LLR",
    params:
        region="{chrom}",
    wrapper:
        "v7.3.0/bio/tabix/query"


rule precomputed_process_chrom_gpn_msa:
    input:
        "results/precomputed/GPN-MSA_LLR/chrom/{chrom}.tsv",
    output:
        "results/precomputed/GPN-MSA_LLR/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                new_columns=COORDINATES + ["score"],
                dtypes={"chrom": str, "score": pl.Float32},
            ).write_parquet(output[0])
        )


rule precomputed_process_chrom_cadd:
    input:
        "results/precomputed/CADD/chrom/{chrom}.tsv",
    output:
        "results/precomputed/CADD/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                columns=[0, 1, 2, 3, 4],
                new_columns=COORDINATES + ["score"],
                dtypes={"chrom": str, "score": pl.Float32},
            ).write_parquet(output[0])
        )


rule precomputed_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        expand("results/precomputed/{{model}}/chrom/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/dataset/{dataset}/features/{model}.parquet",
    wildcard_constraints:
        model="CADD|GPN-MSA_LLR",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0], columns=COORDINATES)
        preds = pl.concat(
            [
                pl.read_parquet(path).join(V, on=COORDINATES, how="inner")
                for path in tqdm(input[1:])
            ]
        )
        V = V.join(preds, on=COORDINATES, how="left")
        V.select("score").write_parquet(output[0])
