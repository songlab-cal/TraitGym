evo2_versions = list(config["evo2"].keys())

n_shards = 6
shards = np.arange(n_shards)


# rule evo2_run_vep:
#    input:
#        "results/dataset/{dataset}/test.parquet",
#        "results/genome.fa.gz",
#    output:
#        "results/dataset/{dataset}/features/evo2_{version}.parquet",
#    wildcard_constraints:
#        version="|".join(evo2_versions),
#    params:
#        window_size=lambda wildcards: config["evo2"][wildcards.version]["window_size"],
#        model_path=lambda wildcards: config["evo2"][wildcards.version]["model_path"],
#        per_device_batch_size=lambda wildcards: config["evo2"][wildcards.version]["per_device_batch_size"],
#    threads:
#        8
#    priority: 999
#    shell:
#        """
#        python \
#        workflow/scripts/run_vep_evo2.py {input} {params.window_size} \
#        {params.model_path} {output} --is_file --dataloader_num_workers 8 \
#        --per_device_batch_size {params.per_device_batch_size} \
#        """


rule evo2_merge_shards:
    input:
        expand(
            "results/dataset/{{dataset}}/features/shard/{shard}/evo2_40b.parquet",
            shard=shards,
        ),
    output:
        "results/dataset/{dataset}/features/evo2_40b.parquet",
    run:
        df = pd.concat([pd.read_parquet(path) for path in input])
        df.to_parquet(output[0], index=False)


rule evo2_extract_llr:
    input:
        "results/dataset/{dataset}/features/evo2_{version}.parquet",
    output:
        "results/dataset/{dataset}/features/evo2_{version}_LLR.parquet",
    wildcard_constraints:
        version="|".join(evo2_versions),
    run:
        (
            pl.read_parquet(input[0], columns=["llr"])
            .rename({"llr": "score"})
            .write_parquet(output[0])
        )


rule evo2_extract_embeddings:
    input:
        "results/dataset/{dataset}/features/evo2_{version}.parquet",
    output:
        "results/dataset/{dataset}/features/evo2_{version}_Embeddings.parquet",
    wildcard_constraints:
        version="|".join(evo2_versions),
    run:
        df = pl.read_parquet(input[0])
        print(df)
        df = df[:, 1:]
        print(df)
        df.write_parquet(output[0])


# near ZRS enhancer
rule evo2_logits:
    input:
        "results/genome.fa.gz",
    output:
        "results/evo2/logits/ZRS.parquet",
    shell:
        """
        python \
        workflow/scripts/evo2_logits.py evo2_40b {input} 7 156789931 156798123 {output}
        """


rule make_probs_wig:
    input:
        "results/evo2/logits/{region}.parquet",
    output:
        temp(expand("results/evo2/probs_wig/{{region}}/{nuc}.wig", nuc=NUCLEOTIDES)),
    run:
        V = pl.read_parquet(input[0])
        assert V["chrom"].n_unique() == 1
        chrom = V["chrom"][0]
        V = V.with_columns(
            pl.DataFrame(softmax(V.select(NUCLEOTIDES), axis=1), schema=NUCLEOTIDES)
        )
        V = V.with_columns(
            entropy=entropy(V.select(NUCLEOTIDES), base=2, axis=1)
        ).with_columns((pl.col(NUCLEOTIDES) * (2 - pl.col("entropy"))))
        for nuc, path in zip(NUCLEOTIDES, output):
            with open(path, "w") as f:
                f.write(f"variableStep chrom=chr{chrom}\n")
            with open(path, "ab") as f:
                V.select(["pos", nuc]).write_csv(
                    f,
                    separator="\t",
                    include_header=False,
                    float_precision=2,
                )


rule make_chrom_sizes:
    input:
        "results/genome.fa.gz",
    output:
        "results/chrom.sizes",
    run:
        intervals = Genome(input[0], subset_chroms=CHROMS).get_all_intervals()
        intervals.chrom = "chr" + intervals.chrom
        intervals.to_csv(
            output[0],
            sep="\t",
            index=False,
            header=False,
            columns=["chrom", "end"],
        )


rule wigToBigWig:
    input:
        "{anything}.wig",
        "results/chrom.sizes",
    output:
        "{anything}.bw",
    shell:
        "wigToBigWig {input} {output} -keepAllChromosomes -fixedSummaries"


rule bigwig_done:
    input:
        expand("{{anything}}/{nuc}.bw", nuc=NUCLEOTIDES),
    output:
        touch("{anything}/bigwig.done"),
