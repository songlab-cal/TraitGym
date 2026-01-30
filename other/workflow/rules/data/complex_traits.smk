rule gwas_download:
    output:
        temp("results/gwas/UKBB_94traits_release1.1.tar.gz"),
        "results/gwas/raw/release1.1/UKBB_94traits_release1.bed.gz",
        "results/gwas/raw/release1.1/UKBB_94traits_release1.cols",
        "results/gwas/raw/release1.1/UKBB_94traits_release1_regions.bed.gz",
        "results/gwas/raw/release1.1/UKBB_94traits_release1_regions.cols",
    params:
        directory("results/gwas/raw"),
    shell:
        """
        wget -O {output[0]} https://www.dropbox.com/s/cdsdgwxkxkcq8cn/UKBB_94traits_release1.1.tar.gz?dl=1 &&
        mkdir -p {params} &&
        tar -xzvf {output[0]} -C {params}
        """


rule gwas_process_main_file:
    input:
        "results/gwas/raw/release1.1/UKBB_94traits_release1.bed.gz",
    output:
        "results/gwas/main_file.parquet",
    run:
        V = (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 2, 5, 6, 10, 11, 14, 15, 17, 21, 22],
                new_columns=[
                    "chrom",
                    "pos",
                    "ref",
                    "alt",
                    "method",
                    "trait",
                    "beta_marginal",
                    "se_marginal",
                    "pip",
                    "LD_HWE",
                    "LD_SV",
                ],
                schema_overrides={"column_3": float},
            )
            .with_columns(pl.col("pos").cast(int))
            .filter(~pl.col("LD_HWE"), ~pl.col("LD_SV"))
            .with_columns((pl.col("beta_marginal") / pl.col("se_marginal")).alias("z"))
        )
        V = (
            V.with_columns(
                p=2 * stats.norm.sf(abs(V["z"])),
            )
            # when PIP > 0.9, manually override as 0.5 when not genome-wide significant
            # (0.5 so it's excluded from both positive and negative set)
            .with_columns(
                pl.when(pl.col("pip") > 0.9, pl.col("p") > 5e-8)
                .then(pl.lit(0.5))
                .otherwise(pl.col("pip"))
                .alias("pip")
            ).select(["chrom", "pos", "ref", "alt", "trait", "method", "pip"])
        )
        print(V)
        V.write_parquet(output[0])


rule gwas_process_secondary_file:
    input:
        "results/gwas/raw/release1.1/UKBB_94traits_release1_regions.bed.gz",
    output:
        "results/gwas/secondary_file.parquet",
    run:
        V = (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[4, 6, 7, 8],
                new_columns=["trait", "variant", "success_finemap", "success_susie"],
            )
            .filter(pl.col("success_finemap"), pl.col("success_susie"))
            .drop("success_finemap", "success_susie")
            .with_columns(
                pl.col("variant")
                .str.split_exact(":", 3)
                .struct.rename_fields(COORDINATES)
            )
            .with_columns(
                pl.col("variant").struct.field("chrom"),
                pl.col("variant").struct.field("pos").cast(int),
                pl.col("variant").struct.field("ref"),
                pl.col("variant").struct.field("alt"),
            )
            .drop("variant")
            .select(["chrom", "pos", "ref", "alt", "trait"])
            .with_columns(pl.lit(0.0).alias("pip"))
        )
        print(V)
        V = pl.concat(
            [
                V.with_columns(pl.lit("SUSIE").alias("method")),
                V.with_columns(pl.lit("FINEMAP").alias("method")),
            ]
        ).select(["chrom", "pos", "ref", "alt", "trait", "method", "pip"])
        print(V)
        V.write_parquet(output[0])


rule gwas_process:
    input:
        "results/gwas/main_file.parquet",
        "results/gwas/secondary_file.parquet",
        "results/genome.fa.gz",
    output:
        "results/gwas/processed.parquet",
    run:
        V = (
            pl.concat([pl.read_parquet(input[0]), pl.read_parquet(input[1])])
            .unique(
                ["chrom", "pos", "ref", "alt", "trait", "method"],
                keep="first",
                maintain_order=True,
            )
            .group_by(["chrom", "pos", "ref", "alt", "trait"])
            .agg(
                pl.mean("pip"),
                (pl.max("pip") - pl.min("pip")).alias("pip_diff"),
                pl.count().alias("pip_n"),
            )
            .filter(pl.col("pip_n") == 2, pl.col("pip_diff") < 0.05)
            .drop("pip_n", "pip_diff")
        )
        print(V)
        V = (
            V.with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(pl.col("trait"))
                .otherwise(pl.lit(None))
                .alias("trait")
            )
            .group_by(COORDINATES)
            .agg(pl.max("pip"), pl.col("trait").drop_nulls().unique())
            .with_columns(pl.col("trait").list.sort().list.join(","))
            .to_pandas()
        )
        print(V)
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_snp(V)
        print(V.shape)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        print(V.shape)
        genome = Genome(input[2])
        V = check_ref_alt(V, genome)
        print(V.shape)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule complex_traits_dataset:
    input:
        "results/gwas/processed.parquet",
        "results/ldscore/UKBB.EUR.ldscore.annot_with_cre.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/complex_traits_matched_{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        V = (
            pl.read_parquet(input[0])
            .with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(True)
                .when(pl.col("pip") < 0.01)
                .then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
            .to_pandas()
        )

        annot = pd.read_parquet(input[1])
        V = V.merge(annot, on=COORDINATES, how="inner")

        V = V[V.consequence.isin(TARGET_CONSEQUENCES)]

        V["start"] = V.pos - 1
        V["end"] = V.pos

        tss = pd.read_parquet(input[2], columns=["chrom", "start", "end"])

        V = (
            bf.closest(V, tss)
            .rename(columns={"distance": "tss_dist"})
            .drop(columns=["start", "end", "chrom_", "start_", "end_"])
        )

        match_features = ["maf", "ld_score", "tss_dist"]

        consequences = V[V.label].consequence.unique()
        V_cs = []
        for c in consequences:
            print(c)
            V_c = V[V.consequence == c].copy()
            for f in match_features:
                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(
                    V_c[f].values.reshape(-1, 1)
                )
            print(V_c.label.value_counts())
            V_c = match_columns_k(
                V_c, "label", [f"{f}_scaled" for f in match_features], k
            )
            V_c["match_group"] = c + "_" + V_c.match_group.astype(str)
            print(V_c.label.value_counts())
            print(V_c.groupby("label")[match_features].median())
            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
            V_cs.append(V_c)
        V = pd.concat(V_cs, ignore_index=True)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule dataset_subset_trait:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/{trait}.parquet",
    wildcard_constraints:
        trait="|".join(select_gwas_traits),
    run:
        V = pd.read_parquet(input[0])
        V.trait = V.trait.str.split(",")
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        V = V[(~V.label) | (V.trait.apply(lambda x: wildcards.trait in x))]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_disease:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/disease.parquet",
    run:
        V = pd.read_parquet(input[0])
        V.trait = V.trait.str.split(",").apply(set)
        target_size = len(V[V.match_group == V.match_group.iloc[0]])

        y = set(config["complex_traits_disease"])

        V = V[(~V.label) | (V.trait.apply(lambda x: len(x & y) > 0))]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V)
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_non_disease:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/non_disease.parquet",
    run:
        V = pd.read_parquet(input[0])
        V.trait = V.trait.str.split(",").apply(set)
        target_size = len(V[V.match_group == V.match_group.iloc[0]])

        y = set(config["complex_traits_disease"])

        V = V[(~V.label) | (V.trait.apply(lambda x: len(x & y) == 0))]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V)
        V[COORDINATES].to_parquet(output[0], index=False)


rule complex_traits_all_dataset:
    input:
        "results/gwas/processed.parquet",
        "results/ldscore/UKBB.EUR.ldscore.annot_with_cre.parquet",
    output:
        "results/dataset/complex_traits_all/test.parquet",
    run:
        V = (
            pl.read_parquet(input[0])
            .with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(True)
                .when(pl.col("pip") < 0.01)
                .then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
            .to_pandas()
        )

        annot = pd.read_parquet(input[1])
        V = V.merge(annot, on=COORDINATES, how="inner")

        V = V[V.consequence.isin(TARGET_CONSEQUENCES)]
        V_pos = V[V.label]
        V = V[V.consequence.isin(V_pos.consequence.unique())]
        V = V[V.chrom.isin(V_pos.chrom.unique())]
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule complex_traits_all_subset_maf_match:
    input:
        "results/dataset/complex_traits_all/test.parquet",
    output:
        "results/dataset/complex_traits_all/subset/maf_match.parquet",
    run:
        V = pd.read_parquet(input[0])
        print(V)
        n_bins = 100
        bins = np.linspace(0, 0.5, n_bins + 1)
        V["maf_bin"] = pd.cut(V.maf, bins=bins, labels=False)
        V_pos = V.query("label")
        V_neg = V.query("not label")
        pos_hist = V_pos.maf_bin.value_counts().sort_index().values
        neg_hist = V_neg.maf_bin.value_counts().sort_index().values
        pos_dist = pos_hist / len(V_pos)
        pos_dist_ratio_to_max = pos_dist / pos_dist.max()
        neg_hist_max = neg_hist[pos_hist.argmax()]
        upper_bound = np.floor(neg_hist_max * pos_dist_ratio_to_max)
        downsample = (neg_hist / upper_bound).min()
        target_n_samples = np.floor(upper_bound * downsample).astype(int)
        V_neg_matched = pd.concat(
            [
                V_neg[V_neg.maf_bin == i].sample(
                    target_n_samples[i], replace=False, random_state=42
                )
                for i in range(n_bins)
            ]
        )
        V = pd.concat([V_pos, V_neg_matched])
        V = sort_variants(V)
        print(V)
        V[COORDINATES].to_parquet(output[0], index=False)


# gene matched
rule complex_dataset_v22:
    input:
        "results/gwas/processed.parquet",
        "results/ldscore/UKBB.EUR.ldscore.annot_with_cre.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/complex_traits_v22_matched_{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        V = (
            pl.read_parquet(input[0])
            .with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(True)
                .when(pl.col("pip") < 0.01)
                .then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
            .to_pandas()
        )

        annot = pd.read_parquet(input[1])
        V = V.merge(annot, on=COORDINATES, how="inner")

        V = V[V.consequence.isin(TARGET_CONSEQUENCES)]

        V["start"] = V.pos - 1
        V["end"] = V.pos

        tss = pd.read_parquet(input[2])

        V = (
            bf.closest(V, tss)
            .rename(
                columns={
                    "distance": "tss_dist",
                    "gene_id_": "gene",
                }
            )
            .drop(columns=["start", "end", "chrom_", "start_", "end_"])
        )

        match_features = ["maf", "ld_score", "tss_dist"]

        consequences = V[V.label].consequence.unique()
        V_cs = []
        for c in consequences:
            print(c)
            V_c = V[V.consequence == c].copy()
            for f in match_features:
                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(
                    V_c[f].values.reshape(-1, 1)
                )
            print(V_c.label.value_counts())
            V_c = match_columns_k_gene(
                V_c, "label", [f"{f}_scaled" for f in match_features], k
            )
            V_c["match_group"] = c + "_" + V_c.match_group.astype(str)
            print(V_c.label.value_counts())
            print(V_c.groupby("label")[match_features].median())
            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
            V_cs.append(V_c)
        V = pd.concat(V_cs, ignore_index=True)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule dataset_subset_pip:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/pip_{pip}.parquet",
    run:
        V = pd.read_parquet(input[0])
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        V = V[(~V.label) | (V.pip > float(wildcards.pip))]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_pleiotropy:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/pleiotropy_{pleiotropy,yes|no}.parquet",
    run:
        V = pd.read_parquet(input[0])
        V["n_traits"] = V.trait.str.split(",").apply(len)
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        if wildcards.pleiotropy == "yes":
            V = V[(~V.label) | (V.n_traits > 1)]
        else:
            V = V[(~V.label) | (V.n_traits == 1)]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)
