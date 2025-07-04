rule run_vep_alphagenome:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/features/AlphaGenome.parquet",
    shell:
        "python workflow/scripts/vep_alphagenome.py {input} {output}"


rule run_vep_alphagenome_v2:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/AlphaGenome_v2.parquet",
    threads:
        workflow.cores
    shell:
        "python workflow/scripts/vep_alphagenome_v2.py {input} {output} --num_workers {threads}"