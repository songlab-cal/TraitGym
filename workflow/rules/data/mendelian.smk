rule merge_clinvar_omim:
    input:
        "results/clinvar/all.annot_with_cre.parquet",
        "results/omim/variants.annot_with_cre.parquet",
    output:
        "results/mendelian/variants.parquet",
    run:
        V_clinvar = pd.read_parquet(input[0]).query('consequence == "missense_variant"')
        V_clinvar["source"] = "ClinVar"
        V_omim = pd.read_parquet(input[1])
        V_omim["source"] = "OMIM"
        V = pd.concat([V_clinvar, V_omim], ignore_index=True)
        V = V.drop_duplicates(COORDINATES)
        V = sort_variants(V)
        V.to_parquet(output[0], index=False)


rule mendelian_dataset:
    input:
        "results/mendelian/variants.annot_MAF.parquet",
        "results/gnomad/common.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/mendelian_matched_{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        pos = pd.read_parquet(input[0])
        pos.maf = pos.maf.fillna(0)
        pos = pos[pos.maf < 0.1 / 100]
        pos = pos.drop(columns=["maf"])
        pos["label"] = True
        neg = pd.read_parquet(input[1])
        neg["label"] = False
        neg["source"] = "gnomAD"
        V = pd.concat([pos, neg], ignore_index=True)
        assert len(V) == len(V.drop_duplicates(COORDINATES))

        V["start"] = V.pos - 1
        V["end"] = V.pos
        tss = pd.read_parquet(input[2], columns=["chrom", "start", "end"])
        V = bf.closest(V, tss).rename(columns={
            "distance": "tss_dist"
        }).drop(columns=["start", "end", "chrom_", "start_", "end_"])

        match_features = ["tss_dist"]

        consequences = V[V.label].consequence.unique()
        V_cs = []
        for c in consequences:
            print(c)
            V_c = V[V.consequence == c].copy()
            for f in match_features:
                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(V_c[f].values.reshape(-1, 1))
            print(V_c.label.value_counts())
            V_c = match_columns_k(V_c, "label", [f"{f}_scaled" for f in match_features], k)
            V_c["match_group"] = c + "_" + V_c.match_group.astype(str)
            print(V_c.label.value_counts())
            print(V_c.groupby("label")[match_features].median())
            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
            V_cs.append(V_c)
        V = pd.concat(V_cs, ignore_index=True)

        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule mendelian_subset_trait:
    input:
        "results/dataset/mendelian_matched_{k}/test.parquet",
    output:
        "results/dataset/mendelian_matched_{k}/subset/{t}.parquet",
    wildcard_constraints:
        t="|".join(select_omim_traits),
    run:
        V = pd.read_parquet(input[0])
        target_size = 1 + int(wildcards.k)
        V = V[(~V.label) | ((V.source=="OMIM") & (V.OMIM == f"MIM {wildcards.t}"))]
        match_group_size = V.match_group.value_counts() 
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        V[COORDINATES].to_parquet(output[0], index=False)
