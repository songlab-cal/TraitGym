rule gnomad_download:
    output:
        "results/gnomad/rare_and_common.annot.parquet",
    run:
        (
            pl.read_parquet(
                "hf://datasets/songlab/gnomad/test.parquet",
                columns=COORDINATES + ["label", "consequence"],
            )
            .with_columns(pl.col("consequence") + "_variant")
            .write_parquet(output[0])
        )


rule gnomad_split_by_label:
    input:
        "results/gnomad/rare_and_common.annot_with_cre.parquet",
    output:
        "results/gnomad/rare.parquet",
        "results/gnomad/common.parquet",
    run:
        V = pl.read_parquet(input[0])
        V.filter(pl.col("label")).drop("label").write_parquet(output[0])
        V.filter(~pl.col("label")).drop("label").write_parquet(output[1])
