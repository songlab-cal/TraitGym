rule run_classifier:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        lambda wildcards: expand("results/dataset/{{dataset}}/features/{features}.parquet", features=config["feature_sets"][wildcards.feature_set]),
    output:
        "results/dataset/{dataset}/preds/{subset}/{feature_set}.{classifier,LogisticRegression|RidgeRegression|RandomForest|XGBoost|PCALogisticRegression|FeatureSelectionLogisticRegression|BestFeature}.{split_mode,chrom|odd_even|pos_quarter}.parquet",
    threads:
        lambda wildcards: 1 if wildcards.classifier == "BestFeature" else workflow.cores
    run:
        2 + 2
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        all_features = []
        for features, path in zip(config["feature_sets"][wildcards.feature_set], input[2:]):
            df = pd.read_parquet(path)
            df.columns = [f"{features}_{col}" for col in df.columns]
            all_features += df.columns.tolist()
            V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        print(V)

        mask_train_list = []
        
        if wildcards.split_mode == "chrom":
            for chrom in V.chrom.unique():
                mask_train = V.chrom != chrom
                mask_train_list.append(mask_train)
        elif wildcards.split_mode == "odd_even":
            for chroms in ODD_EVEN_CHROMS:
                mask_train = V.chrom.isin(chroms)
                mask_train_list.append(mask_train)
        elif wildcards.split_mode == "pos_quarter":
            pos_subsets = np.array_split(np.unique(V.pos), 4)
            for pos_subset in pos_subsets:
                mask_train = V.pos.isin(pos_subset)
                mask_train_list.append(mask_train)

        for mask_train in tqdm(mask_train_list):
            mask_test = ~mask_train
            V.loc[mask_test, "score"] = train_predict(
                V[mask_train], V[mask_test], all_features,
                classifier_map[wildcards.classifier]
            )

        V[["score"]].to_parquet(output[0], index=False)


rule subset_classifier:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/all/{model}.parquet",
    output:
        "results/dataset/{dataset}/preds/{subset}/{model}.subset_from_all.parquet",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = pd.read_parquet(input[2])
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        V[["score"]].to_parquet(output[0], index=False)


rule unsupervised_pred:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/features/{features}.parquet",
    output:
        "results/dataset/{dataset}/preds/{subset}/{features}.{sign,plus|minus}.{feature}.parquet",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = (
            pd.read_parquet(input[2], columns=[wildcards.feature])
            .rename(columns={wildcards.feature: "score"})
        )
        if wildcards.sign == "minus":
            df.score = -df.score
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        assert V.score.isna().sum() == 0
        V[["score"]].to_parquet(output[0], index=False)


rule eval_unsupervised_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/features/{features}.parquet",
    output:
        "results/dataset/{dataset}/unsupervised_metrics/{subset}/{features}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = pd.read_parquet(input[2])
        features = df.columns
        df = df.fillna(df.mean())
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        metric = average_precision_score
        metric_name = "AUPRC"
        res = []
        for f in tqdm(features):
            score_plus = metric(V.label, V[f])
            score_minus = metric(V.label, -V[f])
            score = max(score_plus, score_minus)
            sign = "plus" if score_plus > score_minus else "minus"
            res.append([score, f, sign])
        res = pd.DataFrame(res, columns=[metric_name, "feature", "sign"])
        res = res.sort_values(metric_name, ascending=False)
        res.to_csv(output[0], index=False)


rule classifier_coefficients:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        lambda wildcards: expand("results/dataset/{{dataset}}/features/{features}.parquet", features=config["feature_sets"][wildcards.feature_set]),
    output:
        "results/dataset/{dataset}/coefficients/{subset}/{feature_set}.csv",
    threads:
        workflow.cores
    run:
        2 + 2
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        all_features = []
        for features, path in zip(config["feature_sets"][wildcards.feature_set], input[2:]):
            df = pd.read_parquet(path)
            df.columns = [f"{features}_{col}" for col in df.columns]
            all_features += df.columns.tolist()
            V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        clf = train_lasso_logistic_regression(V[all_features], V.label, V.chrom)
        linear = clf.best_estimator_.named_steps["linear"]
        res = pd.DataFrame({
            "feature": all_features,
            "coef": linear.coef_[0],
        }).sort_values("coef", ascending=False, key=abs)
        res.to_csv(output[0], index=False)


#rule merge_metrics:
#    input:
#        lambda wildcards: expand(
#            "results/dataset/{{dataset}}/metrics/{model}.csv",
#            model=config["dataset_eval_models"][wildcards.dataset]
#        )
#    output:
#        "results/dataset/{dataset}/merged_metrics.csv",
#    run:
#        dfs = [pd.read_csv(f) for f in input]
#        df = pd.concat(dfs, axis=0)
#        df = df.sort_values(df.columns[-1], ascending=False)
#        print(df)
#        df.to_csv(output[0], index=False)
#
#
#rule plot_metrics:
#    input:
#        "results/dataset/{dataset}/test.parquet",
#        "results/dataset/{dataset}/merged_metrics.csv",
#    output:
#        "results/dataset/{dataset}/plot.pdf",
#    run:
#        V = pd.read_parquet(input[0])
#        n_pos, n_neg = V.label.sum(), len(V) - V.label.sum()
#        res = pd.read_csv(input[1])
#        metric = res.columns[-1]
#        if metric == "AUROC":
#            baseline = 0.5
#        elif metric == "AUPRC":
#            baseline = n_pos / (n_pos + n_neg)
#        plt.figure(figsize=(2,2))
#        g = sns.barplot(
#            data=res,
#            y="Model",
#            x=metric,
#            color="C0",
#        )
#        sns.despine()
#        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
#        g.set(xlim=baseline, ylabel="")
#        title = f"{wildcards.dataset.split('/')[-1]}\n{sample_size}"
#        plt.title(title)
#        for bar, model in zip(g.patches, res.Model):
#            text = f'{bar.get_width():.3f}'
#
#            g.text(
#                max(bar.get_width(), baseline),  # X position, here at the end of the bar
#                bar.get_y() + bar.get_height()/2,  # Y position, in the middle of the bar
#                text,  # Text to be displayed, formatted to 3 decimal places
#                va='center'  # Vertical alignment
#            )
#        plt.savefig(output[0], bbox_inches="tight")


def accuracy_score(y_true, y_pred):
    """Check if the highest score corresponds to the true label."""
    y_true = y_true.values
    y_pred = y_pred.values
    assert sum(y_true) == 1
    return float(y_true[np.argmax(y_pred)])


def spearman_score(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def odds_ratio_score(y_true, y_pred, threshold_n=30):
    from scipy.stats import fisher_exact
    import polars as pl

    df = pl.DataFrame({"label": y_true, "score": -y_pred})
    negative_set = df.filter(~pl.col("label")).sort("score")
    threshold = negative_set[threshold_n]["score"]
    group_counts = (
        df.group_by(["label", pl.col("score") <= threshold]).len()
        .sort(["label", "score"])["len"].to_numpy().reshape((2,2))
    )
    return fisher_exact(group_counts, alternative='greater')[0]


def odds_ratio_score_v2(y_true, y_pred, threshold_q=0.05):
    from scipy.stats import fisher_exact
    import polars as pl

    df = pl.DataFrame({"label": y_true, "score": -y_pred})
    threshold = df.select(pl.col("score").quantile(threshold_q))["score"]
    group_counts = (
        df.group_by(["label", pl.col("score") <= threshold]).len()
        .sort(["label", "score"])["len"].to_numpy().reshape((2,2))
    )
    return fisher_exact(group_counts, alternative='greater')[0]


metric_mapping = {
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
    "Accuracy": accuracy_score,
    "Spearman": spearman_score,
    "OR": odds_ratio_score,
    "OR_v2": odds_ratio_score_v2,
    "OR_10%": lambda y_true, y_pred: odds_ratio_score_v2(y_true, y_pred, threshold_q=0.1),
}
BLOCKS = [
    "chrom",
    "gene",
    "match_group",
    "element",
]


rule get_metric:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/{metric}/{subset}/{model}.csv",
    wildcard_constraints:
        metric="|".join(metric_mapping.keys()),
    run:
        metric_name = wildcards.metric
        metric = metric_mapping[metric_name]
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]

        # let's resample within each class
        V = pl.DataFrame(V)
        n_bootstraps = 100
        V = V.select(["label", "score"])

        def resample(V, seed):
            V_pos = V.filter(pl.col("label"))
            V_pos = V_pos.sample(len(V_pos), with_replacement=True, seed=seed)
            V_neg = V.filter(~pl.col("label"))
            V_neg = V_neg.sample(len(V_neg), with_replacement=True, seed=seed)
            return pl.concat([V_pos, V_neg])

        V_bs = [resample(V, i) for i in tqdm(range(n_bootstraps))]
        se = pl.Series([metric(V_b["label"], V_b["score"]) for V_b in tqdm(V_bs)]).std()
        res = pd.DataFrame({
            "Model": [wildcards.model],
            metric_name: [metric(V["label"], V["score"])],
            "se": [se],
        })
        print(res)
        res.to_csv(output[0], index=False)


rule get_metric_by_block:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/{metric}_by_{block}/{subset}/{model}.csv",
    wildcard_constraints:
        metric="|".join(metric_mapping.keys()),
        block="|".join(BLOCKS),
    run:
        metric_name = wildcards.metric
        metric = metric_mapping[metric_name]
        block = wildcards.block
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        res = []
        for b in V[block].unique():
            V_b = V[V[block] == b]
            res.append([b, len(V_b), wildcards.model, metric(V_b.label, V_b.score)])
        res = pd.DataFrame(res, columns=[block, "n", "Model", metric_name])
        res.to_csv(output[0], index=False)


rule get_metric_by_block_weighted_average:
    input:
        "results/dataset/{dataset}/{metric}_by_{block}/{subset}/{model}.csv",
    output:
        "results/dataset/{dataset}/{metric}_by_{block}_weighted_average/{subset}/{model}.csv",
    wildcard_constraints:
        metric="|".join(metric_mapping.keys()),
        block="|".join(BLOCKS),
    run:
        res = pl.read_csv(input[0])
        metric = res.columns[-1]

        def stat(df):
            x = df[metric]
            weight = df["n"] / df["n"].sum()
            return (x * weight).sum()

        res = pl.DataFrame({
            "model": [wildcards.model],
            "metric": [metric],
            "score": [stat(res)],
            "se": [bootstrap_se(res, stat)],
        })
        res.write_csv(output[0])

