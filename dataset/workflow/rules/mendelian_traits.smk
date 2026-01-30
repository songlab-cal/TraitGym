rule mendelian_traits_positives:
    input:
        "results/clinvar/omim.parquet",
        "results/smedley_et_al/variants.parquet",
        "results/hgmd/variants.parquet",
        "results/gnomad/all.parquet",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/mendelian_traits/positives.parquet",
    run:
        gnomad = pl.read_parquet(input[3], columns=COORDINATES + ["AF"])
        V = (
            pl.concat(
                [
                    pl.read_parquet(input[0]).with_columns(source=pl.lit("clinvar")),
                    pl.read_parquet(input[1]).with_columns(
                        source=pl.lit("smedley_et_al")
                    ),
                    pl.read_parquet(input[2]).with_columns(source=pl.lit("hgmd")),
                ],
                how="diagonal_relaxed",
            )
            # reflects the order in which they were concatenated, from higher to lower
            # priority: clinvar, smedley et al, hgmd
            .unique(COORDINATES, keep="first", maintain_order=True)
            .join(gnomad, on=COORDINATES, how="left")
            # variants not in gnomAD are assigned AF = 0
            .with_columns(pl.col("AF").fill_null(0))
            .filter(pl.col("AF") < config["mendelian_traits"]["AF_threshold"])
            .sort(COORDINATES)
        )
        results = []
        for path, chrom in zip(input.consequences, CHROMS):
            chrom_variants = V.filter(pl.col("chrom") == chrom).lazy()
            consequences_lf = pl.scan_parquet(path)
            joined = chrom_variants.join(
                consequences_lf,
                on=COORDINATES,
                how="left",
                maintain_order="left",
            ).collect(engine="streaming")
            results.append(joined)
        pl.concat(results).write_parquet(output[0])


rule mendelian_traits_full_consequence_counts:
    input:
        "results/mendelian_traits/positives.parquet",
    output:
        "results/mendelian_traits/full_consequence_counts.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .group_by(["source", "consequence"])
            .agg(pl.count())
            .sort(["source", "count"], descending=True)
            .write_parquet(output[0])
        )


rule mendelian_traits_dataset_all:
    input:
        "results/mendelian_traits/positives.parquet",
        "results/gnomad/common.parquet",
        "results/intervals/exon.parquet",
        "results/intervals/tss.parquet",
    output:
        "results/dataset/mendelian_traits_all/test.parquet",
    run:
        exon = pl.read_parquet(input[2])
        tss = pl.read_parquet(input[3])
        V = (
            pl.concat(
                [
                    pl.read_parquet(input[0]).with_columns(label=pl.lit(True)),
                    pl.read_parquet(input[1]).with_columns(label=pl.lit(False)),
                ],
                how="diagonal_relaxed",
            )
            .filter(~pl.col("consequence").is_in(config["exclude_consequences"]))
            # order is important, tss_proximal overrides exon_proximal
            .pipe(add_exon, exon, config["exon_proximal_dist"])
            .pipe(add_tss, tss, config["tss_proximal_dist"])
        )
        consequence_final_pos = V.filter("label")["consequence_final"].unique()
        V = V.filter(pl.col("consequence_final").is_in(consequence_final_pos))
        consequence_to_group = {
            c: group
            for group, consequences in config["consequence_groups"].items()
            for c in consequences
        }
        V = V.with_columns(
            pl.col("consequence_final")
            .replace(consequence_to_group)
            .alias("consequence_group")
        )
        V.sort(COORDINATES).write_parquet(output[0])


rule mendelian_traits_dataset_matched:
    input:
        "results/dataset/mendelian_traits_all/test.parquet",
    output:
        "results/dataset/mendelian_traits_matched_{k}/test.parquet",
    run:
        V = pl.read_parquet(input[0])
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                ["tss_dist", "exon_dist"],
                ["chrom", "consequence_final"],
                int(wildcards.k),
            ).write_parquet(output[0])
        )


rule mendelian_traits_matched_feature_performance:
    input:
        "results/dataset/mendelian_traits_matched_{k}/test.parquet",
    output:
        "results/feature_performance/mendelian_traits_matched_{k}.parquet",
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


#
# rule mendelian_traits_dataset_match_gene:
#     input:
#         "results/mendelian_traits/unique_additional_features.parquet",
#     output:
#         "results/dataset/mendelian_traits_match_gene_matched_{k,\d+}/test.parquet",
#     run:
#         V = pl.read_parquet(input[0])
#         V = match_features(
#             V.filter(pl.col("label")),
#             V.filter(~pl.col("label")),
#             ["AF", "tss_dist", "exon_dist"],
#             ["chrom", "consequence", "tss_closest_gene_id", "exon_closest_gene_id"],
#             int(wildcards.k),
#         )
#         V.sort(COORDINATES).write_parquet(output[0])
#
#
