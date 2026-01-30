rule alphagenome_run_vep:
    input:
        lambda wc: config["datasets"][wc.dataset],
    output:
        "results/alphagenome/{dataset}/AlphaGenome_L2.parquet",
    threads: workflow.cores
    shell:
        "python workflow/scripts/vep_alphagenome.py {input} {output} --num_workers {threads}"


rule alphagenome_aggregate_assay:
    input:
        "results/alphagenome/{dataset}/AlphaGenome_L2.parquet",
    output:
        "results/features/{dataset}/AlphaGenome_L2_{operation,max|mean|L2}.parquet",
    run:
        operation = wildcards.operation
        df = pd.read_parquet(input[0])
        col_assay = df.columns.str.split("-").str[0]
        assays = ["all"] + col_assay.unique().tolist()
        result = {}
        for assay in assays:
            df_assay = df if assay == "all" else df.loc[:, col_assay == assay]
            if operation == "max":
                result[assay] = df_assay.max(axis=1)
            elif operation == "mean":
                result[assay] = df_assay.mean(axis=1)
            elif operation == "L2":
                result[assay] = np.linalg.norm(df_assay.values, axis=1, ord=2)
        pd.DataFrame(result).to_parquet(output[0], index=False)
