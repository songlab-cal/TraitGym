rule hgmd_download:
    output:
        temp("results/hgmd/variants.csv"),
    shell:
        """
        mkdir -p results/hgmd && cd results/hgmd && \
        wget https://sei-files.s3.amazonaws.com/pathogenic_mutations.tar.gz && \
        tar -xzvf pathogenic_mutations.tar.gz && \
        mv pathogenic_mutations/sei_data/hgmd.raw.matrix.csv variants.csv && \
        rm -rf pathogenic_mutations pathogenic_mutations.tar.gz
        """


rule hgmd_process:
    input:
        "results/hgmd/variants.csv",
        "results/genome.fa.gz",
    output:
        "results/hgmd/variants.parquet",
    run:
        V = pd.read_csv(
            input[0], usecols=["CHROM_hg19", "POS_hg19", "REF", "ALT", "phenotype"]
        ).rename(
            columns={
                "CHROM_hg19": "chrom",
                "POS_hg19": "pos",
                "REF": "ref",
                "ALT": "alt",
                "phenotype": "trait",
            }
        )
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_chroms(V)
        V = filter_snp(V)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        genome = Genome(input[1])
        V = check_ref_alt(V, genome)
        V = sort_variants(V)
        V.to_parquet(output[0], index=False)
