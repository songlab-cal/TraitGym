rule specieslm_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/SpeciesLM_LLR.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_llr_specieslm.py {input} \
        {config[specieslm][window_size]} {config[specieslm][model_path]} {output} \
        --is_file --dataloader_num_workers 8 \
        --per_device_batch_size {config[specieslm][per_device_batch_size]}
        """


rule specieslm_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/SpeciesLM_Embeddings.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_embeddings_specieslm.py {input} \
        {config[specieslm][window_size]} {config[specieslm][model_path]} {output} \
        --is_file --dataloader_num_workers 8 \
        --per_device_batch_size {config[specieslm][per_device_batch_size]}
        """
