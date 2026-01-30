from sklearn.metrics import average_precision_score


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
        rows = []
        labels = V["label"]
        rows.append(
            {
                "subset": "global",
                "metric": "AUPRC",
                "score": average_precision_score(labels, score),
                "n_pos": labels.sum(),
                "n_neg": len(labels) - labels.sum(),
            }
        )
        for group in V["consequence_group"].unique().sort():
            mask = V["consequence_group"] == group
            labels_subset = V.filter(mask)["label"]
            rows.append(
                {
                    "subset": group,
                    "metric": "AUPRC",
                    "score": average_precision_score(labels_subset, score.filter(mask)),
                    "n_pos": labels_subset.sum(),
                    "n_neg": len(labels_subset) - labels_subset.sum(),
                }
            )
        pl.DataFrame(rows).write_parquet(output[0])
