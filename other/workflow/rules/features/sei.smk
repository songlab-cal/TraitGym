# first, upload a vcf to https://hb.flatironinstitute.org/sei
# download results
# extract the sequence class scores file


rule sei_process:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/sei/{dataset}/sequence-class-scores.tsv.gz",
    output:
        "results/dataset/{dataset}/features/Sei.parquet",
    run:
        V = pd.read_parquet(input[0])
        original_cols = V.columns
        df = pd.read_csv(input[1], sep="\t")
        df.columns = df.columns.str.replace(" ", "_")
        print(df)
        assert df.ref_match.all()
        df.chrom = df.chrom.str.replace("chr", "")
        df = df.drop(columns=["ref_match", "contains_unk", "id", "strand"])
        print(df)
        V = V.merge(df, how="left", on=COORDINATES).drop(columns=original_cols)
        print(V)
        V.to_parquet(output[0], index=False)
