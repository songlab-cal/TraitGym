clinvar_url_vcf = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_{config['clinvar_release']}.vcf.gz"
clinvar_url_tbi = clinvar_url_vcf + ".tbi"


rule clinvar_download:
    output:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    shell:
        "wget {clinvar_url_vcf} -O {output[0]} && wget {clinvar_url_tbi} -O {output[1]}"


rule clinvar_process:
    input:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    output:
        "results/clinvar/all.parquet",
    run:
        from cyvcf2 import VCF

        rows = []
        for variant in VCF(input[0]):
            if variant.INFO.get("CLNVC") != "single_nucleotide_variant":
                continue
            if len(variant.ALT) != 1:
                continue
            cln_sig = variant.INFO.get("CLNSIG")
            if cln_sig != "Pathogenic":
                continue
            review_status = variant.INFO.get("CLNREVSTAT")
            if review_status not in ["reviewed_by_expert_panel", "practice_guideline"]:
                continue
            rows.append([variant.CHROM, variant.POS, variant.REF, variant.ALT[0]])

        V = pd.DataFrame(rows, columns=COORDINATES)
        V = filter_snp(V)
        V = filter_chroms(V)
        V.to_parquet(output[0], index=False)
