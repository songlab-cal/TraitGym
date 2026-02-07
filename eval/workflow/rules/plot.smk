rule plot_model_comparison:
    input:
        metrics=lambda wc: expand(
            "results/metrics/{dataset}/{model}.parquet",
            dataset=wc.dataset,
            model=config["evaluate_models"][wc.dataset],
        ),
    output:
        "results/plots/model_comparison/{dataset}.svg",
    run:
        plots_config = config.get("plots", {})
        models_config = plots_config.get("models", {})
        subsets_config = plots_config.get("subsets", [])
        subset_order, subset_aliases = get_subset_order_and_aliases(subsets_config)

        models = config["evaluate_models"][wildcards.dataset]
        dfs = []
        for path, model in zip(input.metrics, models):
            df = pl.read_parquet(path).with_columns(pl.lit(model).alias("model"))
            dfs.append(df)
        metrics = pl.concat(dfs)

        metrics = metrics.filter(pl.col("metric") == "AUPRC")

        # Get subsets present in data, ordered by config
        available_subsets = set(metrics["subset"].unique().to_list())
        subsets = [s for s in subset_order if s in available_subsets]

        subset_info = (
            metrics.group_by("subset")
            .agg(pl.col("n_pos").first(), pl.col("n_neg").first())
            .to_pandas()
            .set_index("subset")
        )

        n_cols = 5
        n_rows = (len(subsets) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(20, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        for idx, subset in enumerate(subsets):
            ax = axes[idx]
            subset_data = metrics.filter(pl.col("subset") == subset).to_pandas()
            subset_data = subset_data.sort_values("score", ascending=True)

            plot_model_bars(ax, subset_data, models_config)
            ax.set_xlabel("AUPRC")

            n_pos = subset_info.loc[subset, "n_pos"]
            n_neg = subset_info.loc[subset, "n_neg"]
            subset_alias = subset_aliases.get(subset, subset)
            ax.set_title(f"{subset_alias}\n(n={n_pos} vs. {n_neg})")

        for idx in range(len(subsets), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output[0])
        plt.close()


rule plot_model_comparison_combined:
    input:
        metrics=[
            f"results/metrics/{dataset}/{model}.parquet"
            for dataset in config["evaluate_models"]
            for model in config["evaluate_models"][dataset]
        ],
    output:
        "results/plots/model_comparison/combined.svg",
    run:
        plots_config = config.get("plots", {})
        models_config = plots_config.get("models", {})
        subsets_config = plots_config.get("subsets", [])
        datasets_config = plots_config.get("datasets", {})
        subset_order, subset_aliases = get_subset_order_and_aliases(subsets_config)

        # Load all metrics with dataset info
        all_dfs = []
        for dataset in config["evaluate_models"]:
            for model in config["evaluate_models"][dataset]:
                path = f"results/metrics/{dataset}/{model}.parquet"
                df = pl.read_parquet(path).with_columns(
                    pl.lit(model).alias("model"),
                    pl.lit(dataset).alias("dataset"),
                )
                all_dfs.append(df)
        metrics = pl.concat(all_dfs)
        metrics = metrics.filter(pl.col("metric") == "AUPRC")

        # Get subsets and datasets
        available_subsets = set(metrics["subset"].unique().to_list())
        subsets = [s for s in subset_order if s in available_subsets]
        datasets = list(config["evaluate_models"].keys())

        n_rows = len(subsets)
        n_cols = len(datasets)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False
        )

        for row_idx, subset in enumerate(subsets):
            for col_idx, dataset in enumerate(datasets):
                ax = axes[row_idx, col_idx]
                subset_data = metrics.filter(
                    (pl.col("subset") == subset) & (pl.col("dataset") == dataset)
                ).to_pandas()

                if len(subset_data) == 0:
                    ax.set_visible(False)
                    continue

                subset_data = subset_data.sort_values("score", ascending=True)
                plot_model_bars(ax, subset_data, models_config, show_labels=True)

                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel("AUPRC")

                    # Row labels (subset alias) on the right margin
                if col_idx == n_cols - 1:
                    subset_alias = subset_aliases.get(subset, subset)
                    ax.annotate(
                        subset_alias,
                        xy=(1.02, 0.5),
                        xycoords="axes fraction",
                        fontsize=10,
                        fontweight="bold",
                        ha="left",
                        va="center",
                    )

                    # Column labels (dataset alias) on top margin
                if row_idx == 0:
                    dataset_alias = datasets_config.get(dataset, {}).get(
                        "alias", dataset
                    )
                    ax.set_title(dataset_alias, fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)  # Make room for row labels
        plt.savefig(output[0])
        plt.close()


rule plot_model_comparison_combined_ensemble:
    input:
        metrics=[
            f"results/metrics/{dataset}/{model}.parquet"
            for dataset in config["evaluate_models_ensemble"]
            for model in config["evaluate_models_ensemble"][dataset]
        ],
    output:
        "results/plots/model_comparison/combined_ensemble.svg",
    run:
        plots_config = config.get("plots", {})
        models_config = plots_config.get("models", {})
        subsets_config = plots_config.get("subsets", [])
        datasets_config = plots_config.get("datasets", {})
        subset_order, subset_aliases = get_subset_order_and_aliases(subsets_config)

        # Load all metrics with dataset info
        all_dfs = []
        for dataset in config["evaluate_models_ensemble"]:
            for model in config["evaluate_models_ensemble"][dataset]:
                path = f"results/metrics/{dataset}/{model}.parquet"
                df = pl.read_parquet(path).with_columns(
                    pl.lit(model).alias("model"),
                    pl.lit(dataset).alias("dataset"),
                )
                all_dfs.append(df)
        metrics = pl.concat(all_dfs)
        metrics = metrics.filter(pl.col("metric") == "AUPRC")

        # Get subsets and datasets
        available_subsets = set(metrics["subset"].unique().to_list())
        subsets = [s for s in subset_order if s in available_subsets]
        datasets = list(config["evaluate_models_ensemble"].keys())

        n_rows = len(subsets)
        n_cols = len(datasets)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False
        )

        for row_idx, subset in enumerate(subsets):
            for col_idx, dataset in enumerate(datasets):
                ax = axes[row_idx, col_idx]
                subset_data = metrics.filter(
                    (pl.col("subset") == subset) & (pl.col("dataset") == dataset)
                ).to_pandas()

                if len(subset_data) == 0:
                    ax.set_visible(False)
                    continue

                subset_data = subset_data.sort_values("score", ascending=True)
                plot_model_bars(ax, subset_data, models_config, show_labels=True)

                # Only show x-label on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel("AUPRC")

                    # Row labels (subset alias) on the right margin
                if col_idx == n_cols - 1:
                    subset_alias = subset_aliases.get(subset, subset)
                    ax.annotate(
                        subset_alias,
                        xy=(1.02, 0.5),
                        xycoords="axes fraction",
                        fontsize=10,
                        fontweight="bold",
                        ha="left",
                        va="center",
                    )

                    # Column labels (dataset alias) on top margin
                if row_idx == 0:
                    dataset_alias = datasets_config.get(dataset, {}).get(
                        "alias", dataset
                    )
                    ax.set_title(dataset_alias, fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)  # Make room for row labels
        plt.savefig(output[0])
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
