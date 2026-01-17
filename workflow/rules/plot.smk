EXPERIMENT1_SUBSETS = ["all"] + config["consequence_subsets"]


rule plot_experiment1:
    input:
        "results/dataset/{dataset}/test.parquet",
        expand(
            "results/dataset/{{dataset}}/{{metric}}/{subset}/{model}.csv",
            subset=EXPERIMENT1_SUBSETS,
            model=config["models_experiment1"],
        ),
    output:
        "results/plots/experiment1/{dataset}/{metric}.svg",
    run:
        pos_prop = pd.read_parquet(input[0], columns=["label"]).label.mean()
        dfs = []
        for subset in EXPERIMENT1_SUBSETS:
            for model in config["models_experiment1"]:
                path = f"results/dataset/{wildcards.dataset}/{wildcards.metric}/{subset}/{model}.csv"
                df = pd.read_csv(path)
                df["subset"] = subset
                df["model"] = model
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        g = sns.catplot(
            data=df,
            x=wildcards.metric,
            y="model",
            col="subset",
            col_wrap=3,
            kind="bar",
            ci="sd",
            palette="viridis",
            sharex=False,
            height=3,
            aspect=1.0,
        )
        if wildcards.metric == "AUPRC":
            baseline = pos_prop
        elif wildcards.metric == "AUROC":
            baseline = 0.5
        g.set(xlim=baseline)
        g.set_titles(
            col_template="{col_name}",
        )
        g.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")


rule plot_histogram:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/preds/all/{model}.parquet",
        expand(
            "results/dataset/{{dataset}}/subset/{subset}.parquet",
            subset=EXPERIMENT1_SUBSETS,
        ),
    output:
        "results/plots/histogram/{dataset}/{model}.svg",
    run:
        V = pd.read_parquet(input[0])
        V["score"] = pd.read_parquet(input[1])["score"]
        dfs = []
        for subset in EXPERIMENT1_SUBSETS:
            if subset == "all":
                df = V.copy()
            else:
                subset_df = pd.read_parquet(
                    f"results/dataset/{wildcards.dataset}/subset/{subset}.parquet"
                )
                df = subset_df.merge(V, on=COORDINATES, how="left")
            df["subset"] = subset
            dfs.append(df[["score", "label", "subset"]])
        df = pd.concat(dfs, ignore_index=True)
        g = sns.displot(
            data=df,
            x="score",
            hue="label",
            col="subset",
            col_wrap=3,
            kind="hist",
            stat="density",
            common_norm=False,
            common_bins=True,
            bins=40,
            height=3,
            aspect=1.0,
            facet_kws={"sharey": False},
        )
        g.set_titles(col_template="{col_name}")
        g.tight_layout()
        plt.savefig(output[0], bbox_inches="tight")
