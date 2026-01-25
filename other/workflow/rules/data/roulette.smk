rule roulette_download:
    output:
        "results/roulette/{chrom}.vcf.gz",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    threads: workflow.cores // 4
    shell:
        "wget http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/{wildcards.chrom}_rate_v5.2_TFBS_correction_all.vcf.gz -O {output}"


rule roulette_process:
    input:
        "results/roulette/{chrom}.vcf.gz",
    output:
        "results/roulette/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    threads: workflow.cores
    run:
        rows = []
        for variant in VCF(input[0]):
            rows.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0],
                    variant.FILTER,
                    variant.INFO.get("MR"),
                ]
            )
        V = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "FILTER", "MR"])
        V.to_parquet(output[0], index=False)


rule roulette_sample:
    input:
        "results/roulette/{chrom}.parquet",
    output:
        "results/roulette/sample/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    threads: workflow.cores
    run:
        filter_out = ["low", "SFS_bump"]
        (
            pd.read_parquet(input[0])
            .query("FILTER not in @filter_out")
            .sample(n=1_000_000, weights="MR", random_state=42)
            .sort_values(COORDINATES)
            .to_parquet(output[0], index=False)
        )


rule roulette_merge:
    input:
        expand("results/roulette/sample/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/roulette/merged.parquet",
    run:
        (pl.concat([pl.read_parquet(f) for f in input]).write_parquet(output[0]))
