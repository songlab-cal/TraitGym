rule gnomad_download:
    output:
        "results/gnomad/all.parquet",
    shell:
        "wget -O {output} https://huggingface.co/datasets/songlab/gnomad-full/resolve/main/test.parquet"


rule gnomad_common:
    input:
        "results/gnomad/all.parquet",
    output:
        "results/gnomad/common.parquet",
    run:
        (
            pl.scan_parquet(input[0])
            .filter(
                pl.col("AN") >= config["gnomad"]["min_AN"],
                pl.col("AF") > config["gnomad"]["AF_threshold_common"],
            )
            .sink_parquet(output[0])
        )
