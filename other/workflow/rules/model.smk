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
        df = pd.read_parquet(input[2], columns=[wildcards.feature]).rename(
            columns={wildcards.feature: "score"}
        )
        if wildcards.sign == "minus":
            df.score = -df.score
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        assert V.score.isna().sum() == 0
        V[["score"]].to_parquet(output[0], index=False)


rule ensemble_ranks:
    input:
        lambda wildcards: expand(
            "results/dataset/{{dataset}}/preds/{{subset}}/{model}.parquet",
            model=config["ensembles"][wildcards.ensemble],
        ),
    output:
        "results/dataset/{dataset}/preds/{subset}/{ensemble}.parquet",
    wildcard_constraints:
        ensemble="|".join(list(config["ensembles"].keys())),
    run:
        def load_score(path: str) -> pd.Series:
            return -(pd.read_parquet(path)["score"].rank(ascending=False))


        df = pd.concat([load_score(path) for path in input], axis=1)
        df = df.sum(axis=1).to_frame("score")
        df.to_parquet(output[0], index=False)


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


metric_mapping = {
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
}


rule get_metric:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/{metric}/{subset}/{model}.csv",
    wildcard_constraints:
        metric="|".join(metric_mapping.keys()),
        subset="|".join(subsets),
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
        res = pd.DataFrame(
            {
                "Model": [wildcards.model],
                metric_name: [metric(V["label"], V["score"])],
                "se": [se],
            }
        )
        print(res)
        res.to_csv(output[0], index=False)
