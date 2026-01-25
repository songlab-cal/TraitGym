rule tss_dist_feature:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/{dataset}/features/TSS_dist.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["start"] = V["pos"] - 1
        V["end"] = V["pos"]
        tss = pd.read_parquet(input[1], columns=["chrom", "start", "end"])
        V = bf.closest(V, tss).rename(columns={"distance": "score"})
        V = sort_variants(V)
        V[["score"]].to_parquet(output[0], index=False)
