rule download_phyloP_100v:
    output:
        "results/conservation/phyloP-100v.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw -O {output}"


rule download_phyloP_241m:
    output:
        "results/conservation/phyloP-241m.bw",
    shell:
        "wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw -O {output}"


rule download_phastCons_43p:
    output:
        "results/conservation/phastCons-43p.bw",
    shell:
        "wget https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig -O {output}"


rule conservation_features:
    input:
        dataset=lambda wc: config["datasets"][wc.dataset],
        bw="results/conservation/{model}.bw",
    output:
        "results/features/{dataset}/{model,phyloP-100v|phyloP-241m|phastCons-43p}.parquet",
    run:
        df = pd.read_parquet(input.dataset)
        bw = pyBigWig.open(input.bw)
        df["score"] = df.progress_apply(
            lambda v: bw.values(f"chr{v.chrom}", v.pos - 1, v.pos)[0], axis=1
        )
        df = df[["score"]].fillna(0)
        df.to_parquet(output[0], index=False)
