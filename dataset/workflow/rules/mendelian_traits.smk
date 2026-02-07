rule mendelian_traits_positives:
    input:
        "results/omim/variants.parquet",
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
                    pl.read_parquet(input[0]).with_columns(source=pl.lit("omim")),
                    pl.read_parquet(input[1]).with_columns(
                        source=pl.lit("smedley_et_al")
                    ),
                    pl.read_parquet(input[2]).with_columns(source=pl.lit("hgmd")),
                ],
                how="diagonal_relaxed",
            )
            # reflects the order in which they were concatenated, from higher to lower
            # priority: omim, smedley et al, hgmd
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


rule mendelian_traits_dataset_all:
    input:
        "results/mendelian_traits/positives.parquet",
        "results/gnomad/common.parquet",
        "results/intervals/exon.parquet",
        "results/intervals/tss.parquet",
    output:
        "results/dataset/mendelian_traits_all/test.parquet",
    run:
        V = pl.concat(
            [
                pl.read_parquet(input[0]).with_columns(label=pl.lit(True)),
                pl.read_parquet(input[1]).with_columns(label=pl.lit(False)),
            ],
            how="diagonal_relaxed",
        )
        build_dataset(
            V,
            pl.read_parquet(input[2]),
            pl.read_parquet(input[3]),
            config["exclude_consequences"],
            config["exon_proximal_dist"],
            config["tss_proximal_dist"],
            config["consequence_groups"],
        ).write_parquet(output[0])


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
