# torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')


rule hyenadna_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/HyenaDNA_LLR.parquet",
    threads: workflow.cores
    priority: 19
    shell:
        """
        python workflow/scripts/run_vep_llr_hyenadna.py \
        {input} {config[hyenadna][model_path]} {output} \
        --is_file --dataloader_num_workers {threads} --per_device_batch_size 16
        """


rule hyenadna_run_vep_inner_products:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/HyenaDNA_InnerProducts.parquet",
    threads: workflow.cores
    priority: 19
    shell:
        """
        python workflow/scripts/run_vep_inner_products_hyenadna.py \
        {input} {config[hyenadna][model_path]} {output} \
        --is_file --dataloader_num_workers {threads} --per_device_batch_size 16
        """


rule hyenadna_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/HyenaDNA_Embeddings.parquet",
    threads: workflow.cores
    priority: 19
    shell:
        """
        python workflow/scripts/run_vep_embeddings_hyenadna.py \
        {input} {config[hyenadna][model_path]} {output} \
        --is_file --dataloader_num_workers {threads} --per_device_batch_size 16
        """
