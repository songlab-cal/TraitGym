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
        "results/plots/experiment1/{dataset}/{metric}.png",
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
