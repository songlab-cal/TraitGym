rule caduceus_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Caduceus_LLR.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.ss.run_vep {input} {config[caduceus][window_size]} \
        {config[caduceus][model_path]} {output} --is_file \
        --per_device_batch_size {config[caduceus][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule caduceus_run_vep_inner_products:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Caduceus_InnerProducts.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.ss.run_vep_inner_products {input} {config[caduceus][window_size]} \
        {config[caduceus][model_path]} {output} --is_file \
        --per_device_batch_size {config[caduceus][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule caduceus_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Caduceus_Embeddings.parquet",
    threads: workflow.cores
    shell:
        """
        python \
        -m gpn.ss.run_vep_embeddings {input} {config[caduceus][window_size]} \
        {config[caduceus][model_path]} {output} --is_file \
        --per_device_batch_size {config[caduceus][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """
