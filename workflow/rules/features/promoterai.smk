rule promoterai_process:
    input:
        "results/promoterai/promoterAI_tss500.tsv.gz",
    output:
        "results/promoterai/scores.parquet",
    run:
        (
            pl.read_csv(input[0], separator="\t", columns=COORDINATES + ["promoterAI"])
            .with_columns(
                pl.col("chrom").str.replace("chr", "", literal=True),
                pl.col("promoterAI").abs().alias("score"),
            )
            .drop("promoterAI")
            .group_by(COORDINATES)
            .agg(pl.max("score"))
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule promoterai_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/promoterai/scores.parquet",
    output:
        "results/dataset/{dataset}/features/PromoterAI.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .join(pl.read_parquet(input[1]), on=COORDINATES, how="left")
            .select("score")
            # no score mean no effect... this is a hack for convenience
            # TODO: figure how to handle this properly (or just exclude PromoterAI)
            .fill_null(0)
            .write_parquet(output[0])
        )
