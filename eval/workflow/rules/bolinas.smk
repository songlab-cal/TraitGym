rule bolinas_download_dataset:
    output:
        "results/bolinas/datasets/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(BOLINAS_DATASETS),
    params:
        repo_id=lambda wildcards: config["bolinas"][wildcards.dataset]["repo_id"],
        revision=lambda wildcards: config["bolinas"][wildcards.dataset]["revision"],
    run:
        dfs = []
        for split in ["train", "test"]:
            path = hf_hub_download(
                repo_id=params.repo_id,
                filename=f"{split}.parquet",
                repo_type="dataset",
                revision=params.revision,
            )
            dfs.append(pl.read_parquet(path).with_columns(pl.lit(split).alias("split")))
        pl.concat(dfs, how="vertical").write_parquet(output[0])


rule bolinas_pack_predictions:
    input:
        dataset="results/bolinas/datasets/{dataset}.parquet",
        features="results/features/{dataset}/GPN-Star-{model}_cLLR.parquet",
    output:
        "results/bolinas/predictions/{dataset}/GPN-Star-{model}.parquet",
    wildcard_constraints:
        dataset="|".join(BOLINAS_DATASETS),
        model="|".join(GPN_STAR_MODELS),
    run:
        ids = pl.read_parquet(
            input.dataset, columns=["chrom", "pos", "ref", "alt", "split"]
        )
        feats = pl.read_parquet(input.features)
        pl.concat([ids, feats], how="horizontal").write_parquet(output[0])


rule gpn_star_bolinas_all:
    input:
        [
            f"results/bolinas/predictions/{dataset}/GPN-Star-{model}.parquet"
            for dataset in BOLINAS_DATASETS
            for model in GPN_STAR_MODELS
        ],
