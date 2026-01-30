rule enformer_metadata:
    output:
        "results/metadata/Enformer.csv",
    run:
        import grelu.resources

        model = grelu.resources.load_model(project="enformer", model_name="human")
        metadata = pd.DataFrame(model.data_params["tasks"])
        metadata.to_csv(output[0], index=False)


rule borzoi_metadata:
    output:
        "results/metadata/Borzoi.csv",
    run:
        import grelu.resources

        model = grelu.resources.load_model(project="borzoi", model_name="human_fold0")
        metadata = pd.DataFrame(model.data_params["tasks"])
        metadata.to_csv(output[0], index=False)


rule run_vep_enformer:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Enformer_L2.parquet",
    priority: 101
    threads: workflow.cores
    shell:
        """
        python \
        workflow/scripts/vep_enformer_borzoi.py {input} enformer human {output} \
        --per_device_batch_size {config[enformer][batch_size]} --dataloader_num_workers {threads} --is_file
        """


# torchrun --nproc_per_node $(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{{print NF}}') \


rule run_vep_borzoi:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/genome.fa.gz",
    output:
        "results/dataset/{dataset}/features/Borzoi_L2.parquet",
    threads: workflow.cores
    priority: 100
    shell:
        """
        python \
        workflow/scripts/vep_enformer_borzoi.py {input} borzoi human_fold0 {output} \
        --per_device_batch_size {config[borzoi][batch_size]} --dataloader_num_workers {threads} --is_file
        """


rule enformer_borzoi_top_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/features/{model}_L2.parquet",
        "results/metadata/{model}.csv",
    output:
        "results/interpretation/top_features/{dataset}/{model,Enformer|Borzoi}_L2.csv",
    run:
        V = pd.read_parquet(input[0])
        features = pd.read_parquet(input[1])
        metadata = pd.read_csv(input[2], usecols=["name", "assay", "description"])
        balanced = V.label.sum() == len(V) // 2
        metric = roc_auc_score if balanced else average_precision_score
        metric_name = "AUROC" if balanced else "AUPRC"
        res = []
        for f in tqdm(features.columns):
            res.append([metric(V.label, features[f]), f])
        res = pd.DataFrame(res, columns=[metric_name, "feature"])
        res = res.merge(metadata, left_on="feature", right_on="name")
        res = res.sort_values(metric_name, ascending=False)
        res.to_csv(output[0], index=False)


rule grelu_aggregate_assay:
    input:
        "results/dataset/{dataset}/features/{model}_L2.parquet",
        "results/metadata/{model}.csv",
    output:
        "results/dataset/{dataset}/features/{model,Enformer|Borzoi}_L2_L{norm_ord}.parquet",
    run:
        df = pd.read_parquet(input[0])
        metadata = pd.read_csv(input[1])
        assays = metadata.assay.unique().tolist() + ["all"]
        if wildcards.norm_ord == "1":
            norm_ord = 1
        elif wildcards.norm_ord == "2":
            norm_ord = 2
        elif wildcards.norm_ord == "inf":
            norm_ord = np.inf
        for assay in assays:
            df_assay = (
                df[metadata[metadata.assay == assay].name] if assay != "all" else df
            )
            df[assay] = np.linalg.norm(df_assay, axis=1, ord=norm_ord)
        df[assays].to_parquet(output[0], index=False)


# rule grelu_features:
#    output:
#        "results/features/{dataset}/{model,Enformer|Borzoi}_L2.parquet",
#    threads:
#        workflow.cores
#    run:
#        import grelu.resources
#        import grelu.variant
#
#        V = load_dataset(wildcards.dataset, split="test").to_pandas()
#        V.chrom = "chr" + V.chrom
#        print(V)
#
#        if wildcards.model == "Enformer":
#            project, model_name = "enformer", "human"
#        elif wildcards.model == "Borzoi":
#            project, model_name = "borzoi", "human_fold0"
#        model = grelu.resources.load_model(project=project, model_name=model_name)
#
#        chrom_sizes = (
#            bf.fetch_chromsizes("hg38").to_frame().reset_index()
#            .rename(columns={"index": "chrom"})
#        )
#        L = model.data_params["train_seq_len"]
#        V["start"] = V.pos - 1 - L // 2
#        V["end"] = V.start + L
#        V = pd.merge(V, chrom_sizes, on="chrom", how="left")
#        V["within_bounds"] = (V.start >= 0) & (V.end <= V.length)
#        print(f"{V.within_bounds.value_counts()=}")
#
#        def log2fc_1p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#            return np.log2(1+x) - np.log2(1+y)
#
#        lfc = grelu.variant.predict_variant_effects(
#            variants=V.loc[V.within_bounds, COORDINATES],
#            model=model,
#            devices=list(range(torch.cuda.device_count())),
#            num_workers=threads,
#            batch_size=8,
#            genome="hg38",
#            compare_func=log2fc_1p,
#            return_ad=False,
#            # Reverse complement the ref/alt predictions and average them.
#            rc=True,
#        )
#        l2_score = np.linalg.norm(lfc, axis=2)
#        columns = model.data_params['tasks']["name"]
#        V.loc[V.within_bounds, columns] = l2_score
#        V[columns].to_parquet(output[0], index=False)
