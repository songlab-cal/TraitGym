rule unsupervised_pred:
    input:
        features="results/features/{dataset}/{features}.parquet",
    output:
        "results/preds/{dataset}/{features}.{sign}.{feature}.parquet",
    wildcard_constraints:
        sign="plus|minus",
    run:
        score = pl.read_parquet(input.features, columns=[wildcards.feature])[
            wildcards.feature
        ]
        if wildcards.sign == "minus":
            score = -score
        score.rename("score").to_frame().write_parquet(output[0])


rule ensemble_ranks:
    input:
        lambda wildcards: expand(
            "results/preds/{{dataset}}/{model}.parquet",
            model=config["ensembles"][wildcards.ensemble],
        ),
    output:
        "results/preds/{dataset}/{ensemble}.parquet",
    wildcard_constraints:
        ensemble="|".join(list(config["ensembles"].keys())),
    run:
        def load_score(path: str) -> pl.Series:
            return -pl.read_parquet(path)["score"].rank(descending=True)


        data = {f"score_{i}": load_score(path) for i, path in enumerate(input)}
        df = pl.DataFrame(data)
        df.sum_horizontal().rename("score").to_frame().write_parquet(output[0])


rule compute_metrics:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/metrics/{dataset}/{model}.parquet",
    run:
        V = pl.read_parquet(input.dataset, columns=["label", "consequence_group"])
        score = pl.read_parquet(input.preds)["score"]
        # Patch: fill NaN with mean for GPN-MSA on mendelian_traits_all if <0.01% NaN
        if wildcards.dataset == "mendelian_traits_all" and wildcards.model.startswith(
            "GPN-MSA"
        ):
            nan_pct = score.null_count() / len(score) * 100
            if nan_pct > 0 and nan_pct < 0.01:
                score = score.fill_null(score.mean())


        def compute_subset_metrics(
            subset_name: str, labels: pl.Series, scores: pl.Series
        ) -> dict:
            return {
                "subset": subset_name,
                "metric": "AUPRC",
                "score": average_precision_score(labels, scores),
                "se": stratified_bootstrap_se(average_precision_score, labels, scores),
                "n_pos": labels.sum(),
                "n_neg": len(labels) - labels.sum(),
            }


        rows = []

        # Per consequence group
        for group in V["consequence_group"].unique().sort():
            mask = V["consequence_group"] == group
            rows.append(
                compute_subset_metrics(
                    group, V.filter(mask)["label"], score.filter(mask)
                )
            )

        pl.DataFrame(rows).write_parquet(output[0])


rule compute_metrics_pairwise:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/metrics_pairwise/{dataset}/{model}.parquet",
    run:
        V = pl.read_parquet(
            input.dataset, columns=["label", "consequence_group", "match_group"]
        )
        score = pl.read_parquet(input.preds)["score"]
        rows = []
        for group in V["consequence_group"].unique().sort():
            mask = V["consequence_group"] == group
            subset = V.filter(mask)
            group_scores = pairwise_accuracy_scores(
                subset["label"], score.filter(mask), subset["match_group"]
            )
            mean, se = iid_mean_se(group_scores)
            rows.append(
                {
                    "subset": group,
                    "metric": "pairwise_accuracy",
                    "score": mean,
                    "se": se,
                    "n_pos": subset["label"].sum(),
                    "n_neg": len(subset) - subset["label"].sum(),
                }
            )
        pl.DataFrame(rows).write_parquet(output[0])


rule compute_metrics_mrr:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        preds="results/preds/{dataset}/{model}.parquet",
    output:
        "results/metrics_mrr/{dataset}/{model}.parquet",
    run:
        V = pl.read_parquet(
            input.dataset, columns=["label", "consequence_group", "match_group"]
        )
        score = pl.read_parquet(input.preds)["score"]
        rows = []
        for group in V["consequence_group"].unique().sort():
            mask = V["consequence_group"] == group
            subset = V.filter(mask)
            group_scores = mean_reciprocal_rank_scores(
                subset["label"], score.filter(mask), subset["match_group"]
            )
            mean, se = iid_mean_se(group_scores)
            rows.append(
                {
                    "subset": group,
                    "metric": "MRR",
                    "score": mean,
                    "se": se,
                    "n_pos": subset["label"].sum(),
                    "n_neg": len(subset) - subset["label"].sum(),
                }
            )
        pl.DataFrame(rows).write_parquet(output[0])
