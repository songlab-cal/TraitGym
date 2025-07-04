# possible columns:
# https://github.com/KalinNonchev/gnomAD_DB/blob/master/gnomad_db/pkgdata/gnomad_columns.yaml
# e.g. 
# - AF # Alternate allele frequency in samples
# - AF_eas # Alternate allele frequency in samples of East Asian ancestry
# - AF_nfe # Alternate allele frequency in XY samples of Non-Finnish European ancestry
# - AF_fin # Alternate allele frequency in XX samples of Finnish ancestry
# - AF_afr # Alternate allele frequency in samples of African/African-American ancestry
# - AF_asj # Alternate allele frequency in samples of Ashkenazi Jewish ancestry


rule download_gnomad_db:
    output:
        directory("results/gnomad_db"),
    run:
        from gnomad_db.database import gnomAD_DB

        url = "https://zenodo.org/record/6818606/files/gnomad_db_v3.1.2.sqlite3.gz?download=1"
        os.makedirs(output[0], exist_ok=True)
        gnomAD_DB.download_and_unzip(url, output[0])


rule maf_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/gnomad_db",
    output:
        "results/dataset/{dataset}/features/M{af_col}.parquet",
    threads:
        workflow.cores
    run:
        from gnomad_db.database import gnomAD_DB

        V = pd.read_parquet(input[0])
        db = gnomAD_DB(input[1], gnomad_version="v3", parallel=True)
        V["score"] = db.get_info_from_df(V, wildcards.af_col)[wildcards.af_col]
        V.score = V.score.where(V.score < 0.5, 1 - V.score)
        V[["score"]].to_parquet(output[0], index=False)


rule annot_maf:
    input:
        "{anything}.parquet",
        "results/gnomad_db",
    output:
        "{anything}.annot_M{af_col}.parquet",
    threads:
        workflow.cores
    run:
        from gnomad_db.database import gnomAD_DB

        V = pd.read_parquet(input[0])
        db = gnomAD_DB(input[1], gnomad_version="v3", parallel=True)
        V["maf"] = db.get_info_from_df(V, wildcards.af_col)[wildcards.af_col]
        V.maf = V.maf.where(V.maf < 0.5, 1 - V.maf)
        V.to_parquet(output[0], index=False)


rule annot_af:
    input:
        "{anything}.parquet",
        "results/gnomad_db",
    output:
        "{anything}.annot_{af_col,AF}.parquet",
    threads:
        workflow.cores
    run:
        from gnomad_db.database import gnomAD_DB

        V = pd.read_parquet(input[0])
        db = gnomAD_DB(input[1], gnomad_version="v3", parallel=True)
        V["AF"] = db.get_info_from_df(V, wildcards.af_col)[wildcards.af_col]
        V["AF"] = V["AF"].fillna(0)
        V.to_parquet(output[0], index=False)
