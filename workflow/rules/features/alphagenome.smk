rule run_vep_alphagenome:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/AlphaGenome_L2.parquet",
    threads:
        workflow.cores
    shell:
        "python workflow/scripts/vep_alphagenome.py {input} {output} --num_workers {threads}"