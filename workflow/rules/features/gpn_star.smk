rule gpn_star_download_model:
    output:
        directory("results/gpn_star/checkpoints/{model}"),
    params:
        repo_id=lambda wildcards: config["gpn_star"][wildcards.model]["repo_id"],
    threads: workflow.cores
    shell:
        "hf download {params.repo_id} --local-dir {output} --max-workers {threads}"


# intermediate output used to obtain both LLR, entropy
rule get_logits:
    input:
        "results/dataset/{dataset}/test.parquet",
        lambda wildcards: config["gpn_star"][wildcards.model]["msa_path"],
        "results/gpn_star/checkpoints/{model}",
    output:
        "results/gpn_star/logits/{dataset}/{model}.parquet",
    params:
        window_size=lambda wildcards: config["gpn_star"][wildcards.model]["window_size"],
    resources:
        # using resources to avoid re-runs when changing batch size
        per_device_batch_size=lambda wildcards: config["gpn_star"][wildcards.model][
            "per_device_batch_size"
        ],
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.star.inference logits \
        {input[0]} {input[1]} {params.window_size} {input[2]} {output} \
        --per_device_batch_size {resources.per_device_batch_size} \
        --dataloader_num_workers {threads} \
        --is_file
        """


rule get_llr_calibrated:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        "results/gpn_star/checkpoints/{model}",
        "results/gpn_star/logits/{dataset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/features/GPN-Star-{model}_LLR.parquet",
    params:
        calibration_path="results/gpn_star/checkpoints/{model}/calibration_table/llr.parquet",
    run:
        from gpn.star.utils import normalize_logits, get_llr

        V = pd.read_parquet(input[0])
        genome = Genome(input[1])
        V["pentanuc"] = V.apply(
            lambda row: genome.get_seq(
                row["chrom"], row["pos"] - 3, row["pos"] + 2
            ).upper(),
            axis=1,
        )
        V["pentanuc_mut"] = V["pentanuc"] + "_" + V["alt"]
        df_calibration = pd.read_parquet(params.calibration_path)
        logits = pd.read_parquet(input[3])
        normalized_logits = normalize_logits(logits)
        V["llr"] = get_llr(normalized_logits, V["ref"], V["alt"])
        V = V.merge(df_calibration, on="pentanuc_mut", how="left")
        V["llr_calibrated"] = V["llr"] - V["llr_neutral_mean"]
        V[["llr", "llr_calibrated"]].to_parquet(output[0], index=False)
