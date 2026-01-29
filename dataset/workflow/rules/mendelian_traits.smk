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
        V.sort(COORDINATES).write_parquet(output[0])


# rule mendelian_traits_add_additional_features:
#     input:
#         "results/mendelian_traits/unique.parquet",
#         "results/intervals/cre.parquet",
#         "results/intervals/exon.parquet",
#         "results/intervals/tss.parquet",
#     output:
#         "results/mendelian_traits/unique_additional_features.parquet",
#     run:
#         V = pl.read_parquet(input[0])
#         V = V.with_columns(original_consequence=pl.col("consequence"))
#         cre = pl.read_parquet(input[1])
#         exon = pl.read_parquet(input[2])
#         tss = pl.read_parquet(input[3])
#         V = add_cre(V, cre)
#         V = add_exon(V, exon)
#         V = add_tss(V, tss)
#         V.write_parquet(output[0])
#
#
# rule mendelian_traits_dataset:
#     input:
#         "results/mendelian_traits/unique_additional_features.parquet",
#     output:
#         "results/dataset/mendelian_traits_matched_{k,\d+}/test.parquet",
#     run:
#         V = pl.read_parquet(input[0])
#         V = match_features(
#             V.filter(pl.col("label")),
#             V.filter(~pl.col("label")),
#             ["AF", "tss_dist", "exon_dist"],
#             ["chrom", "consequence"],
#             int(wildcards.k),
#         )
#         V.sort(COORDINATES).write_parquet(output[0])
#
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
# rule mendelian_traits_all_dataset:
#     input:
#         "results/mendelian_traits/unique_additional_features.parquet",
#     output:
#         "results/dataset/mendelian_traits_all/test.parquet",
#     run:
#         V = pl.read_parquet(input[0])
#         V.sort(COORDINATES).write_parquet(output[0])
#
#
# rule mendelian_traits_legacy_dataset:
#     output:
#         "results/dataset/mendelian_traits_legacy/test.parquet",
#     shell:
#         "wget -O {output} https://huggingface.co/datasets/songlab/TraitGym/resolve/main/mendelian_traits_matched_9/test.parquet"
#
#
# rule mendelian_traits_alt_add_features:
#     input:
#         "results/mendelian_traits/unique.parquet",
#         "results/intervals/cre.parquet",
#         "results/intervals/exon.parquet",
#         "results/intervals/tss.parquet",
#         "results/gnomad/common.annot.parquet",
#     output:
#         "results/mendelian_traits/alt.parquet",
#     run:
#         V = pl.read_parquet(input[0]).filter(pl.col("source") != "roulette")
#         gnomad = (
#             pl.read_parquet(input[4])
#             .filter(pl.col("consequence").is_in(config["consequences"]))
#             .with_columns(source=pl.lit("gnomad"))
#         )
#         V = pl.concat([V, gnomad], how="diagonal_relaxed").sort(COORDINATES)
#
#         V = V.with_columns(original_consequence=pl.col("consequence"))
#         cre = pl.read_parquet(input[1])
#         exon = pl.read_parquet(input[2])
#         tss = pl.read_parquet(input[3])
#         V = add_cre(V, cre)
#         V = add_exon(V, exon)
#         V = add_tss(V, tss)
#         V.write_parquet(output[0])
#
#
# rule mendelian_traits_alt_dataset:
#     input:
#         "results/mendelian_traits/alt.parquet",
#     output:
#         "results/dataset/mendelian_traits_alt_matched_{k,\d+}/test.parquet",
#     run:
#         V = pl.read_parquet(input[0])
#         V = match_features(
#             V.filter(pl.col("label")),
#             V.filter(~pl.col("label")),
#             ["tss_dist", "exon_dist"],
#             ["chrom", "consequence"],
#             int(wildcards.k),
#         )
#         V.sort(COORDINATES).write_parquet(output[0])
#
