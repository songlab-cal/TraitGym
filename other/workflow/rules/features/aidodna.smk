rule aidodna_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/AIDO.DNA_LLR.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_llr_aidodna.py {input} \
        {config[aidodna][window_size]} {config[aidodna][model_path]} {output} \
        --is_file --dataloader_num_workers 8 \
        --per_device_batch_size {config[aidodna][per_device_batch_size]} \
        --n_prefix 1
        """


rule aidodna_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/AIDO.DNA_Embeddings.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_embeddings_aidodna.py {input} \
        {config[aidodna][window_size]} {config[aidodna][model_path]} {output} \
        --is_file --dataloader_num_workers 8 \
        --per_device_batch_size {config[aidodna][per_device_batch_size]}
        """
