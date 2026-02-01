rule ldscore_download:
    output:
        temp("results/ldscore/UKBB.EUR.ldscore.ht"),
    shell:
        "aws s3 cp --no-sign-request --recursive s3://pan-ukb-us-east-1/ld_release/UKBB.EUR.ldscore.ht {output}"


rule ldscore_convert:
    input:
        "results/ldscore/UKBB.EUR.ldscore.ht",
    output:
        "results/ldscore/UKBB.EUR.ldscore.tsv.bgz",
    singularity:
        "docker://hailgenetics/hail:0.2.130.post1-py3.11"
    shell:
        "python3 workflow/scripts/ht2tsv.py {input} {output}"


# $ zcat results/ldscore/UKBB.EUR.ldscore.tsv.bgz | head
# locus   alleles rsid    AF      ld_score
# 1:11063 ["T","G"]       rs561109771     4.7982e-05      5.7386e+00
# 1:13259 ["G","A"]       rs562993331     2.7798e-04      5.0488e+00
# 1:17641 ["G","A"]       rs578081284     8.3096e-04      1.6833e+00
# 1:57222 ["T","C"]       rs576081345     6.5859e-04      2.4759e+00
# 1:58396 ["T","C"]       rs570371753     2.4023e-04      2.9534e+01
# 1:63668 ["G","A"]       rs561430336     2.7728e-05      2.0150e+00
# 1:69569 ["T","C"]       rs2531267       1.8542e-04      2.0772e+00
# 1:79192 ["T","G"]       rs557418932     8.1599e-05      4.3001e+00
# 1:91588 ["G","A"]       rs554639997     1.5868e-04      2.1612e+00


rule ldscore_process:
    input:
        "results/ldscore/UKBB.EUR.ldscore.tsv.bgz",
    output:
        "results/ldscore/UKBB.EUR.ldscore.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                separator="\t",
                columns=["locus", "alleles", "AF", "ld_score"],
            )
            .with_columns(
                pl.col("locus")
                .str.split_exact(":", 1)
                .struct.rename_fields(["chrom", "pos"]),
                pl.col("alleles")
                .str.replace("[", "", literal=True)
                .str.replace("]", "", literal=True)
                .str.replace_all('"', "", literal=True)
                .str.split_exact(",", 1)
                .struct.rename_fields(["ref", "alt"]),
                pl.when(pl.col("AF") < 0.5)
                .then(pl.col("AF"))
                .otherwise(1 - pl.col("AF"))
                .alias("MAF"),
            )
            .with_columns(
                pl.col("locus").struct.field("chrom"),
                pl.col("locus").struct.field("pos").cast(int),
                pl.col("alleles").struct.field("ref"),
                pl.col("alleles").struct.field("alt"),
            )
            .drop(["locus", "alleles"])
            .pipe(filter_snp)
            .select(COORDINATES + ["MAF", "ld_score"])
            .sort(COORDINATES)
            .write_parquet(output[0])
        )
