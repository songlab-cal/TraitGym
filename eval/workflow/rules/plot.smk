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

        # Build lookup dicts from config
        subset_order = [s["name"] for s in subsets_config]
        subset_aliases = {s["name"]: s["alias"] for s in subsets_config}

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
            # Sort by AUPRC descending (best model at top)
            subset_data = subset_data.sort_values("score", ascending=True)

            y_positions = range(len(subset_data))
            colors = [
                models_config.get(m, {}).get("color", f"C{i%10}")
                for i, m in enumerate(subset_data["model"])
            ]
            labels = [
                models_config.get(m, {}).get("alias", m) for m in subset_data["model"]
            ]

            # Create horizontal bar plot
            ax.barh(y_positions, subset_data["score"], color=colors)

            # Add error bars
            ax.errorbar(
                x=subset_data["score"],
                y=y_positions,
                xerr=subset_data["se"],
                fmt="none",
                color="black",
                capsize=0,
            )

            ax.set_yticks(list(y_positions))
            ax.set_yticklabels(labels)
            ax.set_xlabel("AUPRC")

            n_pos = subset_info.loc[subset, "n_pos"]
            n_neg = subset_info.loc[subset, "n_neg"]
            prop_pos = n_pos / (n_pos + n_neg)
            ax.set_xlim(left=prop_pos)
            subset_alias = subset_aliases.get(subset, subset)
            ax.set_title(f"{subset_alias}\n(n={n_pos} vs. {n_neg})")
            sns.despine(ax=ax)

            # Hide unused axes
        for idx in range(len(subsets), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
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
