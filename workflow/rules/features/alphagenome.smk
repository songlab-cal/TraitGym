rule run_vep_alphagenome:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/AlphaGenome.parquet",
    shell:
        "python workflow/scripts/vep_alphagenome.py {input} {output}"
