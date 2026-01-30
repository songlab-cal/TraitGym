rule gpn_star_download_model:
    output:
        directory("results/gpn_star/checkpoints/{model}"),
    wildcard_constraints:
        model="|".join(GPN_STAR_MODELS),
    params:
        repo_id=lambda wildcards: config["gpn_star"][wildcards.model]["repo_id"],
    threads: workflow.cores
    shell:
        "hf download {params.repo_id} --local-dir {output} --max-workers {threads}"


rule gpn_star_get_logits:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        msa_path=lambda wildcards: config["gpn_star"][wildcards.model]["msa_path"],
        checkpoint="results/gpn_star/checkpoints/{model}",
    output:
        "results/gpn_star/logits/{dataset}/{model}.parquet",
    wildcard_constraints:
        model="|".join(GPN_STAR_MODELS),
    params:
        window_size=lambda wildcards: config["gpn_star"][wildcards.model]["window_size"],
    resources:
        per_device_batch_size=lambda wildcards: config["gpn_star"][wildcards.model][
            "per_device_batch_size"
        ],
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.star.inference logits \
        {input.dataset} {input.msa_path} {params.window_size} {input.checkpoint} {output} \
        --per_device_batch_size {resources.per_device_batch_size} \
        --dataloader_num_workers {threads} \
        --is_file
        """


rule gpn_star_get_llr_calibrated:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        genome="results/genome.fa.gz",
        checkpoint="results/gpn_star/checkpoints/{model}",
        logits="results/gpn_star/logits/{dataset}/{model}.parquet",
    output:
        "results/features/{dataset}/GPN-Star-{model}_LLR.parquet",
    wildcard_constraints:
        model="|".join(GPN_STAR_MODELS),
    params:
        calibration_path="results/gpn_star/checkpoints/{model}/calibration_table/llr.parquet",
    run:
        V = pl.read_parquet(input.dataset, columns=["chrom", "pos", "ref", "alt"])
        genome = Genome(input.genome)
        V = V.with_columns(
            pl.struct(["chrom", "pos"])
            .map_elements(
                lambda row: genome.get_seq(
                    row["chrom"], row["pos"] - 3, row["pos"] + 2
                ).upper(),
                return_dtype=pl.Utf8,
            )
            .alias("pentanuc")
        )
        V = V.with_columns(
            (pl.col("pentanuc") + "_" + pl.col("alt")).alias("pentanuc_mut")
        )
        df_calibration = pl.read_parquet(params.calibration_path)
        logits = pl.read_parquet(input.logits)
        normalized_logits = normalize_logits(logits.to_pandas())
        V = V.with_columns(
            pl.Series(
                "llr",
                get_llr(normalized_logits, V["ref"].to_list(), V["alt"].to_list()),
            )
        )
        V = V.join(df_calibration, on="pentanuc_mut", how="left")
        V = V.with_columns(
            (pl.col("llr") - pl.col("llr_neutral_mean")).alias("llr_calibrated")
        )
        V.select(["llr", "llr_calibrated"]).write_parquet(output[0])
