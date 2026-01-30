rule plot_model_comparison:
    input:
        metrics=expand(
            "results/metrics/{{dataset}}/{model}.parquet",
            model=config["models"],
        ),
    output:
        "results/plots/{dataset}/model_comparison.svg",
    run:
        dfs = []
        for path, model in zip(input.metrics, config["models"]):
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
