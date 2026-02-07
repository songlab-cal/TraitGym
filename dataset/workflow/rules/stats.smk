rule mendelian_traits_full_consequence_counts:
    input:
        "results/mendelian_traits/positives.parquet",
    output:
        "results/stats/full_consequence_counts/mendelian_traits.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .group_by(["source", "consequence"])
            .agg(pl.count())
            .sort(["source", "count"], descending=True)
            .write_parquet(output[0])
        )


rule complex_traits_full_consequence_counts:
    input:
        "results/complex_traits/annotated.parquet",
    output:
        "results/stats/full_consequence_counts/complex_traits.parquet",
    run:
        (
            pl.read_parquet(input[0], columns=["label", "consequence"])
            .filter(pl.col("label"))
            .get_column("consequence")
            .value_counts()
            .sort("count", descending=True)
            .write_parquet(output[0])
        )


rule mendelian_traits_matched_feature_performance:
    input:
        "results/dataset/mendelian_traits_matched_{k}/test.parquet",
    output:
        "results/stats/matched_feature_performance/mendelian_traits_matched_{k}.parquet",
    run:
        V = pl.read_parquet(input[0])
        features = ["tss_dist", "exon_dist"]
        rows = []

        for feature in features:
            # Global AUPRC (negative distance: closer = higher score)
            auprc = average_precision_score(V["label"], -V[feature])
            rows.append(
                {
                    "feature": feature,
                    "consequence_final": "all",
                    "auprc": auprc,
                    "n_pos": V["label"].sum(),
                    "n_neg": (~V["label"]).sum(),
                }
            )

            # Per consequence_final
            for consequence in V["consequence_final"].unique().sort():
                subset = V.filter(pl.col("consequence_final") == consequence)
                auprc = average_precision_score(subset["label"], -subset[feature])
                rows.append(
                    {
                        "feature": feature,
                        "consequence_final": consequence,
                        "auprc": auprc,
                        "n_pos": subset["label"].sum(),
                        "n_neg": (~subset["label"]).sum(),
                    }
                )

        pl.DataFrame(rows).write_parquet(output[0])


rule complex_traits_matched_feature_performance:
    input:
        "results/dataset/complex_traits_matched_{k}/test.parquet",
    output:
        "results/stats/matched_feature_performance/complex_traits_matched_{k}.parquet",
    run:
        V = pl.read_parquet(input[0])
        features = ["tss_dist", "exon_dist", "MAF", "ld_score"]
        # Sign: +1 if higher value predicts positive, -1 if lower value predicts positive
        sign = {"tss_dist": -1, "exon_dist": -1, "MAF": 1, "ld_score": -1}
        rows = []

        for feature in features:
            auprc = average_precision_score(V["label"], sign[feature] * V[feature])
            rows.append(
                {
                    "feature": feature,
                    "consequence_final": "all",
                    "auprc": auprc,
                    "n_pos": V["label"].sum(),
                    "n_neg": (~V["label"]).sum(),
                }
            )

            # Per consequence_final
            for consequence in V["consequence_final"].unique().sort():
                subset = V.filter(pl.col("consequence_final") == consequence)
                auprc = average_precision_score(
                    subset["label"], sign[feature] * subset[feature]
                )
                rows.append(
                    {
                        "feature": feature,
                        "consequence_final": consequence,
                        "auprc": auprc,
                        "n_pos": subset["label"].sum(),
                        "n_neg": (~subset["label"]).sum(),
                    }
                )

        pl.DataFrame(rows).write_parquet(output[0])


rule mendelian_traits_full_consequence_counts_to_tex:
    input:
        "results/stats/full_consequence_counts/mendelian_traits.parquet",
    output:
        "results/tex/full_consequence_counts/mendelian_traits.tex",
    run:
        df = pl.read_parquet(input[0])

        lines = [
            r"\begin{tabular}{llr}",
            r"\toprule",
            r"Source & Consequence & Count \\",
            r"\midrule",
        ]

        for row in df.iter_rows(named=True):
            source = row["source"].replace("_", r"\_")
            consequence = row["consequence"].replace("_", r"\_")
            lines.append(f"{source} & {consequence} & {row['count']} " + r"\\")

        lines.extend([r"\bottomrule", r"\end{tabular}"])

        with open(output[0], "w") as f:
            f.write("\n".join(lines) + "\n")


rule complex_traits_full_consequence_counts_to_tex:
    input:
        "results/stats/full_consequence_counts/complex_traits.parquet",
    output:
        "results/tex/full_consequence_counts/complex_traits.tex",
    run:
        df = pl.read_parquet(input[0])

        lines = [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Consequence & Count \\",
            r"\midrule",
        ]

        for row in df.iter_rows(named=True):
            consequence = row["consequence"].replace("_", r"\_")
            lines.append(f"{consequence} & {row['count']} " + r"\\")

        lines.extend([r"\bottomrule", r"\end{tabular}"])

        with open(output[0], "w") as f:
            f.write("\n".join(lines) + "\n")


rule matched_feature_performance_to_tex:
    input:
        "results/stats/matched_feature_performance/{dataset}.parquet",
    output:
        "results/tex/matched_feature_performance/{dataset}.tex",
    run:
        df = pl.read_parquet(input[0]).filter(pl.col("consequence_final") == "all")

        feature_names = {
            "tss_dist": "(-) TSS distance",
            "exon_dist": "(-) Exon distance",
            "MAF": "MAF",
            "ld_score": "(-) LD score",
        }

        lines = [
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Feature & AUPRC \\",
            r"\midrule",
        ]

        for row in df.iter_rows(named=True):
            name = feature_names.get(row["feature"], row["feature"])
            lines.append(f"{name} & {row['auprc']:.3f} " + r"\\")

        lines.extend([r"\bottomrule", r"\end{tabular}"])

        with open(output[0], "w") as f:
            f.write("\n".join(lines) + "\n")
