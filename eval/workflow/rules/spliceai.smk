rule spliceai_extract_chrom:
    input:
        "results/spliceai/spliceai_scores.masked.snv.hg38.vcf.gz",
    output:
        temp("results/precomputed/SpliceAI/chrom/{chrom}.tsv"),
    wildcard_constraints:
        chrom="|".join(CHROMS),
    params:
        region="{chrom}",
    wrapper:
        "v7.3.0/bio/tabix/query"


rule spliceai_to_parquet:
    input:
        "results/precomputed/SpliceAI/chrom/{chrom}.tsv",
    output:
        "results/precomputed/SpliceAI/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_csv(
                input[0],
                has_header=False,
                separator="\t",
                columns=[0, 1, 3, 4, 7],
                new_columns=COORDINATES + ["INFO"],
                schema_overrides={"chrom": str},
            )
            .with_columns(
                pl.col("INFO")
                .str.split("|")
                .list.slice(2, 4)
                .cast(pl.Array(pl.Float32, 4))
                .arr.max()
                .alias("score")
            )
            .drop("INFO")
            .group_by(COORDINATES)
            .agg(pl.max("score"))
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule compute_metrics_spliceai_covered:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        spliceai="results/features/{dataset}/SpliceAI.parquet",
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/metrics_spliceai_covered/{dataset}/{model}.parquet",
    run:
        V = pl.read_parquet(input.dataset, columns=["label", "consequence_group"])
        spliceai = pl.read_parquet(input.spliceai)["score"]
        score = pl.read_parquet(input.preds)["score"]

        mask = (V["consequence_group"] == "splicing") & spliceai.is_not_null()
        labels = V.filter(mask)["label"]
        scores = score.filter(mask)

        pl.DataFrame(
            [
                {
                    "subset": "splicing_spliceai_covered",
                    "metric": "AUPRC",
                    "score": average_precision_score(labels, scores),
                    "se": stratified_bootstrap_se(
                        average_precision_score, labels, scores
                    ),
                    "n_pos": labels.sum(),
                    "n_neg": len(labels) - labels.sum(),
                }
            ]
        ).write_parquet(output[0])


rule plot_spliceai_covered:
    input:
        metrics=[
            f"results/metrics_spliceai_covered/{dataset}/{model}.parquet"
            for dataset in config.get("evaluate_models_spliceai_covered", {})
            for model in config.get("evaluate_models_spliceai_covered", {}).get(
                dataset, []
            )
        ],
    output:
        "results/plots/model_comparison/spliceai_covered.svg",
    run:
        models_config = config.get("plots", {}).get("models", {})
        datasets_config = config.get("plots", {}).get("datasets", {})

        all_dfs = []
        for dataset in config["evaluate_models_spliceai_covered"]:
            for model in config["evaluate_models_spliceai_covered"][dataset]:
                path = f"results/metrics_spliceai_covered/{dataset}/{model}.parquet"
                df = pl.read_parquet(path).with_columns(
                    pl.lit(model).alias("model"),
                    pl.lit(dataset).alias("dataset"),
                )
                all_dfs.append(df)
        metrics = pl.concat(all_dfs)

        datasets = list(config["evaluate_models_spliceai_covered"].keys())
        n_cols = len(datasets)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)

        for col_idx, dataset in enumerate(datasets):
            ax = axes[0, col_idx]
            subset_data = (
                metrics.filter(pl.col("dataset") == dataset)
                .to_pandas()
                .sort_values("score", ascending=True)
            )
            plot_model_bars(ax, subset_data, models_config, show_labels=True)
            ax.set_xlabel("AUPRC")
            dataset_alias = datasets_config.get(dataset, {}).get("alias", dataset)
            n_pos = subset_data["n_pos"].iloc[0]
            n_neg = subset_data["n_neg"].iloc[0]
            ax.set_title(f"{dataset_alias} (n={n_pos} vs. {n_neg})")

        plt.tight_layout()
        plt.savefig(output[0])
        plt.close()
