rule gnomad_download:
    output:
        "results/gnomad/common.annot.parquet",
    run:
        (
            pl.read_parquet(
                "hf://datasets/songlab/gnomad/test.parquet",
                columns=COORDINATES + ["label", "consequence"],
            )
            .filter(~pl.col("label"))
            .with_columns(pl.col("consequence") + "_variant")
            .write_parquet(output[0])
        )
