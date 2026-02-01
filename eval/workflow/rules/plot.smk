rule plot_model_comparison:
    input:
        metrics=lambda wc: expand(
            "results/metrics/{dataset}/{model}.parquet",
            dataset=wc.dataset,
            model=config["evaluate_models"][wc.dataset],
        ),
    output:
        "results/plots/{dataset}/model_comparison.svg",
    run:
        models = config["evaluate_models"][wildcards.dataset]
        dfs = []
        for path, model in zip(input.metrics, models):
            df = pl.read_parquet(path).with_columns(pl.lit(model).alias("model"))
            dfs.append(df)
        metrics = pl.concat(dfs)

        metrics = metrics.filter(pl.col("metric") == "AUPRC")

        subsets = metrics["subset"].unique().sort().to_list()
        if "global" in subsets:
            subsets.remove("global")
            subsets = ["global"] + subsets

        subset_info = (
            metrics.group_by("subset")
            .agg(pl.col("n_pos").first(), pl.col("n_neg").first())
            .to_pandas()
            .set_index("subset")
        )

        g = sns.catplot(
            data=metrics.to_pandas(),
            x="score",
            y="model",
            col="subset",
            col_order=subsets,
            col_wrap=3,
            kind="bar",
            sharex=False,
        )
        g.set_axis_labels("AUPRC", "Model")

        for ax, subset in zip(g.axes.flat, subsets):
            n_pos = subset_info.loc[subset, "n_pos"]
            n_neg = subset_info.loc[subset, "n_neg"]
            prop_pos = n_pos / (n_pos + n_neg)
            ax.set_xlim(left=prop_pos)
            ax.set_title(f"{subset} (n={n_pos} vs. {n_neg})")

        g.savefig(output[0])
        plt.close()


rule plot_score_histogram:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/plots/score_histogram/{dataset}/{model}.svg",
    run:
        dataset = pl.read_parquet(input.dataset, columns=["label", "consequence_group"])
        preds = pl.read_parquet(input.preds, columns=["score"])

        # Combine dataset and predictions (row-aligned)
        df = pl.concat([dataset, preds], how="horizontal")

        # Define consequence_group order for consistent panel arrangement
        consequence_order = df["consequence_group"].unique().sort().to_list()

        g = sns.displot(
            data=df.to_pandas(),
            x="score",
            row="consequence_group",
            row_order=consequence_order,
            hue="label",
            kind="hist",
            stat="density",
            common_norm=False,
            common_bins=True,
            height=3,
            facet_kws={"sharey": False},
        )
        g.set_titles("{row_name}")
        g.set_axis_labels("Prediction Score", "Density")
        g.savefig(output[0])
        plt.close()
