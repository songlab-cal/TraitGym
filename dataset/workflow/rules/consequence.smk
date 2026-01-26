rule consequence_download:
    output:
        "results/consequences/{chrom}.parquet",
    shell:
        "wget -O {output} https://huggingface.co/datasets/songlab/hg38-variant-consequences/resolve/main/{wildcards.chrom}.parquet"
