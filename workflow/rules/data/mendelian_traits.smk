rule mendelian_traits_merge_sources:
    input:
        "results/clinvar/omim.parquet",
        "results/smedley_et_al/variants.parquet",
        "results/hgmd/variants.parquet",
        "results/roulette/merged.parquet",
    output:
        "results/mendelian_traits/all.parquet",
    run:
        (
            pl.concat(
                [
                    pl.read_parquet(input[0]).with_columns(source=pl.lit("clinvar")),
                    pl.read_parquet(input[1]).with_columns(
                        source=pl.lit("smedley_et_al")
                    ),
                    pl.read_parquet(input[2]).with_columns(source=pl.lit("hgmd")),
                    pl.read_parquet(input[3]).with_columns(source=pl.lit("roulette")),
                ],
                how="diagonal_relaxed",
            ).write_parquet(output[0])
        )


rule mendelian_traits_filter_AF:
    input:
        "results/mendelian_traits/all.annot_AF.parquet",
    output:
        "results/mendelian_traits/filt_AF.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("AF") < config["mendelian_traits"]["af_threshold"] / 100)
            .write_parquet(output[0])
        )


rule mendelian_traits_full_consequence_counts:
    input:
        "results/mendelian_traits/filt_AF.annot.parquet",
    output:
        "results/mendelian_traits/consequence_counts.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("source") != "roulette")
            .group_by(["source", "consequence"])
            .agg(pl.count())
            .write_parquet(output[0])
        )


rule mendelian_traits_filter_consequence:
    input:
        "results/mendelian_traits/filt_AF.annot.parquet",
    output:
        "results/mendelian_traits/filt_AF_consequence.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("consequence").is_in(config["consequences"]))
            .write_parquet(output[0])
        )


rule mendelian_traits_drop_duplicates:
    input:
        "results/mendelian_traits/filt_AF_consequence.parquet",
    output:
        "results/mendelian_traits/unique.parquet",
    run:
        (
            pl.read_parquet(input[0])
            # reflects the order in which they were concatenated:
            # clinvar, smedley et al, hgmd, roulette
            .unique(COORDINATES, keep="first", maintain_order=True)
            .with_columns(label=pl.col("source") != "roulette")
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule mendelian_traits_add_additional_features:
    input:
        "results/mendelian_traits/unique.parquet",
        "results/intervals/cre.parquet",
        "results/intervals/exon.parquet",
        "results/intervals/tss.parquet",
    output:
        "results/mendelian_traits/unique_additional_features.parquet",
    run:
        V = pl.read_parquet(input[0])
        V = V.with_columns(original_consequence=pl.col("consequence"))
        cre = pl.read_parquet(input[1])
        exon = pl.read_parquet(input[2])
        tss = pl.read_parquet(input[3])
        V = add_cre(V, cre)
        V = add_exon(V, exon)
        V = add_tss(V, tss)
        V.write_parquet(output[0])


rule mendelian_traits_dataset:
    input:
        "results/mendelian_traits/unique_additional_features.parquet",
    output:
        "results/dataset/mendelian_traits_matched_{k,\d+}/test.parquet",
    run:
        V = pl.read_parquet(input[0])
        V = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            ["AF", "tss_dist", "exon_dist"],
            ["chrom", "consequence"],
            int(wildcards.k),
        )
        V.sort(COORDINATES).write_parquet(output[0])


rule mendelian_traits_dataset_match_gene:
    input:
        "results/mendelian_traits/unique_additional_features.parquet",
    output:
        "results/dataset/mendelian_traits_match_gene_matched_{k,\d+}/test.parquet",
    run:
        V = pl.read_parquet(input[0])
        V = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            ["AF", "tss_dist", "exon_dist"],
            ["chrom", "consequence", "tss_closest_gene_id", "exon_closest_gene_id"],
            int(wildcards.k),
        )
        V.sort(COORDINATES).write_parquet(output[0])


rule mendelian_traits_all_dataset:
    input:
        "results/mendelian_traits/unique_additional_features.parquet",
    output:
        "results/dataset/mendelian_traits_all/test.parquet",
    run:
        V = pl.read_parquet(input[0])
        V.sort(COORDINATES).write_parquet(output[0])
