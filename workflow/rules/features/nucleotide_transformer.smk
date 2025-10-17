# seems to be running out of memory, will not spend more time on this since it's not a priority
# torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}')


rule nucleotide_transformer_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/NucleotideTransformer_LLR.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_llr_nucleotide_transformer.py \
        {input} {config[nucleotide_transformer][model_path]} {output} \
        --is_file --dataloader_num_workers 8 --per_device_batch_size 64
        """


rule nucleotide_transformer_run_vep_inner_products:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/NucleotideTransformer_InnerProducts.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_inner_products_nucleotide_transformer.py \
        {input} {config[nucleotide_transformer][model_path]} {output} \
        --is_file --dataloader_num_workers 8 --per_device_batch_size 32
        """


rule nucleotide_transformer_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/NucleotideTransformer_Embeddings.parquet",
    threads: workflow.cores
    priority: 20
    shell:
        """
        python workflow/scripts/run_vep_embeddings_nucleotide_transformer.py \
        {input} {config[nucleotide_transformer][model_path]} {output} \
        --is_file --dataloader_num_workers 8 --per_device_batch_size 32
        """
