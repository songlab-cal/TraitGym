rule promoterai_process:
    input:
        "results/promoterai/promoterAI_tss500.tsv.gz",
    output:
        "results/promoterai/scores.parquet",
    run:
        (
            pl.read_csv(input[0], separator="\t", columns=COORDINATES + ["promoterAI"])
            .with_columns(
                pl.col("chrom").str.replace("chr", "", literal=True),
                pl.col("promoterAI").abs().alias("score"),
            )
            .drop("promoterAI")
            .group_by(COORDINATES)
            .agg(pl.max("score"))
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule promoterai_features:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        scores="results/promoterai/scores.parquet",
    output:
        "results/features/{dataset}/PromoterAI.parquet",
    run:
        (
            pl.read_parquet(input.dataset, columns=COORDINATES)
            .join(
                pl.read_parquet(input.scores),
                on=COORDINATES,
                how="left",
                maintain_order="left",
            )
            .select("score")
            .write_parquet(output[0])
        )


rule compute_metrics_promoterai_covered:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        promoterai="results/features/{dataset}/PromoterAI.parquet",
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/metrics_promoterai_covered/{dataset}/{model}.parquet",
    run:
        V = pl.read_parquet(input.dataset, columns=["label", "consequence_group"])
        promoterai = pl.read_parquet(input.promoterai)["score"]
        score = pl.read_parquet(input.preds)["score"]

        mask = (V["consequence_group"] == "tss_proximal") & promoterai.is_not_null()
        labels = V.filter(mask)["label"]
        scores = score.filter(mask)

        pl.DataFrame(
            [
                {
                    "subset": "tss_proximal_promoterai_covered",
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


rule plot_promoterai_covered:
    input:
        metrics=[
            f"results/metrics_promoterai_covered/{dataset}/{model}.parquet"
            for dataset in config["evaluate_models_promoterai_covered"]
            for model in config["evaluate_models_promoterai_covered"][dataset]
        ],
    output:
        "results/plots/model_comparison/promoterai_covered.svg",
    run:
        models_config = config.get("plots", {}).get("models", {})
        datasets_config = config.get("plots", {}).get("datasets", {})

        all_dfs = []
        for dataset in config["evaluate_models_promoterai_covered"]:
            for model in config["evaluate_models_promoterai_covered"][dataset]:
                path = f"results/metrics_promoterai_covered/{dataset}/{model}.parquet"
                df = pl.read_parquet(path).with_columns(
                    pl.lit(model).alias("model"),
                    pl.lit(dataset).alias("dataset"),
                )
                all_dfs.append(df)
        metrics = pl.concat(all_dfs)

        datasets = list(config["evaluate_models_promoterai_covered"].keys())
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
