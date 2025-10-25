clinvar_url_vcf = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_{config['clinvar']['release']}.vcf.gz"
clinvar_url_tbi = clinvar_url_vcf + ".tbi"
clinvar_submission_summary_url = config["clinvar"]["submission_summary_url"]


rule clinvar_download:
    output:
        temp("results/clinvar/all.vcf.gz"),
        temp("results/clinvar/all.vcf.gz.tbi"),
    shell:
        "wget {clinvar_url_vcf} -O {output[0]} && wget {clinvar_url_tbi} -O {output[1]}"


rule clinvar_process:
    input:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    output:
        "results/clinvar/all.parquet",
    run:
        rows = []
        for variant in VCF(input[0]):
            if variant.INFO.get("CLNVC") != "single_nucleotide_variant":
                continue
            if len(variant.ALT) != 1:
                continue
            rows.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0],
                    int(variant.ID),
                ]
            )
        V = pd.DataFrame(rows, columns=COORDINATES + ["clinvar_id"])
        V = filter_snp(V)
        V = filter_chroms(V)
        V.to_parquet(output[0], index=False)


rule clinvar_submission_summary_download:
    output:
        "results/clinvar/submission_summary.txt.gz",
    shell:
        "wget {clinvar_submission_summary_url} -O {output}"


rule clinvar_submission_summary_process:
    input:
        "results/clinvar/submission_summary.txt.gz",
    output:
        "results/clinvar/submission_summary.parquet",
    run:
        pd.read_csv(input[0], sep="\t", skiprows=18).to_parquet(output[0], index=False)


rule clinvar_submission_summary_omim:
    input:
        "results/clinvar/submission_summary.parquet",
    output:
        "results/clinvar/submission_summary_omim.parquet",
    run:
        df = (
            pl.read_parquet(
                input[0],
                columns=[
                    "#VariationID",
                    "ClinicalSignificance",
                    "SubmittedPhenotypeInfo",
                    "ReportedPhenotypeInfo",
                    "CollectionMethod",
                    "Submitter",
                ],
            )
            .rename({"#VariationID": "clinvar_id"})
            .filter(
                pl.col("Submitter") == "OMIM",
                pl.col("CollectionMethod") == "literature only",
            )
            .drop(["CollectionMethod", "Submitter"])
        )
        unambiguous_ids = (
            df.group_by("clinvar_id")
            .n_unique()
            .filter(pl.col("ClinicalSignificance") == 1)["clinvar_id"]
        )
        df = df.filter(
            pl.col("clinvar_id").is_in(unambiguous_ids),
            pl.col("ClinicalSignificance") == "Pathogenic",
            ~pl.col("SubmittedPhenotypeInfo").str.contains("RECLASSIFIED"),
        ).drop(["ClinicalSignificance", "SubmittedPhenotypeInfo"])
        df = (
            df.group_by("clinvar_id")
            .agg(pl.col("ReportedPhenotypeInfo").unique())
            .with_columns(pl.col("ReportedPhenotypeInfo").list.join("|").alias("trait"))
            .drop("ReportedPhenotypeInfo")
        )
        df.write_parquet(output[0])


rule clinvar_omim:
    input:
        "results/clinvar/all.parquet",
        "results/clinvar/submission_summary_omim.parquet",
    output:
        "results/clinvar/omim.parquet",
    run:
        V = pl.read_parquet(input[0])
        df = pl.read_parquet(input[1])
        V = V.join(df, on="clinvar_id", how="inner", maintain_order="left")
        V.write_parquet(output[0])
