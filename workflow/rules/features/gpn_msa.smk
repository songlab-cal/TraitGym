rule gpn_msa_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_LLR.parquet",
    threads: workflow.cores
    priority: 103
    shell:
        """
        python \
        -m gpn.msa.inference vep {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


# rule gpn_msa_run_vep_inner_products:
#    input:
#        "results/dataset/{dataset}/test.parquet",
#    output:
#        "results/dataset/{dataset}/features/GPN-MSA_InnerProducts.parquet",
#    threads:
#        workflow.cores
#    priority: 103
#    shell:
#        """
#        python \
#        -m gpn.msa.inference vep_embedding {input} {config[gpn_msa][msa_path]} \
#        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
#        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
#        """


rule gpn_msa_run_vep_influence:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_Influence.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.msa.inference vep_influence {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


rule gpn_msa_run_vep_ref_embed:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_RefEmbed.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.msa.inference vep_ref_embed {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


rule gpn_msa_run_vep_delta_embed:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_DeltaEmbed.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.msa.inference vep_delta_embed {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


rule gpn_msa_run_vep_euclidean_distance:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_EuclideanDistance.parquet",
    threads: workflow.cores
    priority: 103
    shell:
        """
        python \
        -m gpn.msa.inference vep_euclidean_dist {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


rule gpn_msa_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-MSA_Embeddings.parquet",
    threads: workflow.cores
    priority: 103
    shell:
        """
        python \
        -m gpn.msa.inference vep_embeddings {input} {config[gpn_msa][msa_path]} \
        {config[gpn_msa][window_size]} {config[gpn_msa][model_path]} {output} \
        --per_device_batch_size 2048 --dataloader_num_workers {threads} --is_file
        """


rule download_gpnmsa_all:
    output:
        "results/gpnmsa/all.tsv.gz",
        "results/gpnmsa/all.tsv.gz.tbi",
    shell:
        """
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz -O {output[0]} &&
        wget https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main/scores.tsv.bgz.tbi -O {output[1]}
        """


rule gpnmsa_extract_chrom:
    input:
        "results/gpnmsa/all.tsv.gz",
    output:
        temp("results/gpnmsa/chrom/{chrom}.tsv"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    shell:
        "tabix {input} {wildcards.chrom} > {output}"


rule gpnmsa_process_chrom:
    input:
        "results/gpnmsa/chrom/{chrom}.tsv",
    output:
        "results/gpnmsa/chrom/{chrom}.parquet",
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


rule run_vep_gpnmsa_in_memory:
    input:
        "results/dataset/{dataset}/test.parquet",
        expand("results/gpnmsa/chrom/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/dataset/{dataset}/features/gpnmsa.parquet",
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
        V = V.with_columns(-pl.col("score"))  # minus LLR
        print(V)
        V.select("score").write_parquet(output[0])
