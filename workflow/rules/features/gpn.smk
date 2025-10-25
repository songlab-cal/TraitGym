rule gpn_version_run_vep_llr:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_LLR.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_inner_products:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_InnerProducts.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_inner_products {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_euclidean_distance:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_EuclideanDistance2.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_embed_dist {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_euclidean_distances:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_EuclideanDistances.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_embed_dists {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_mean_euclidean_distance:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_MeanEuclideanDistance.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_mean_euclidean_dist {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_cosine_similarity:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_CosineSimilarity.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_cosine_similarity {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_mean_cosine_similarity:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_MeanCosineSimilarity.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_mean_cosine_similarity {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_influence:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_Influence.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_influence {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """


rule gpn_version_run_vep_embeddings:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
        lambda wildcards: config["gpn"][wildcards.version]["model_path"],
    output:
        "results/dataset/{dataset}/features/GPN_{version}_Embeddings.parquet",
    threads: workflow.cores
    priority: 102
    shell:
        """
        python \
        -m gpn.ss.run_vep_embeddings {input[0]} {input[1]} {config[gpn][window_size]} \
        {input[2]} {output} --is_file \
        --per_device_batch_size {config[gpn][per_device_batch_size]} \
        --dataloader_num_workers {threads}
        """
