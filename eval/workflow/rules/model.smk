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
        rows = []
        rows.append(
            {
                "subset": "global",
                "metric": "AUPRC",
                "score": average_precision_score(V["label"], score),
            }
        )
        for group in V["consequence_group"].unique().sort():
            mask = V["consequence_group"] == group
            rows.append(
                {
                    "subset": group,
                    "metric": "AUPRC",
                    "score": average_precision_score(
                        V.filter(mask)["label"], score.filter(mask)
                    ),
                }
            )
        pl.DataFrame(rows).write_parquet(output[0])
