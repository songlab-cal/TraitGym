# Curated regulatory OMIM variants, Table S6 from:
# Smedley, Damian, et al. "A whole-genome analysis framework for effective
# identification of pathogenic regulatory variants in Mendelian disease." The
# American Journal of Human Genetics 99.3 (2016): 595-606.


# just a subset we'll use for interpretation
#omim_traits = [
#    "174500",
#    "306900",
#    "600886",
#]


rule omim_download:
    output:
        temp("results/omim/variants.xslx"),
    shell:
        "wget -O {output} https://ars.els-cdn.com/content/image/1-s2.0-S0002929716302786-mmc2.xlsx"


rule omim_process:
    input:
        "results/omim/variants.xslx",
    output:
        "results/omim/variants.parquet",
    run:
        xls = pd.ExcelFile(input[0])
        sheet_names = xls.sheet_names

        dfs = []
        for variant_type in sheet_names[1:]:
            df = pd.read_excel(input[0], sheet_name=variant_type)
            dfs.append(df)
        V = pd.concat(dfs)
        V = V[["Chr", "Position", "Ref", "Alt", "OMIM"]].rename(columns={
            "Chr": "chrom", "Position": "pos", "Ref": "ref", "Alt": "alt"
        })
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_snp(V)
        V = lift_hg19_to_hg38(V)
        V = V[V.pos != -1]
        print(V)
        V.to_parquet(output[0], index=False)


rule omim_dataset:
    input:
        "results/omim/variants.annot_with_cre.parquet",
        "results/gnomad/common.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/omim_matched_{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        pos = pd.read_parquet(input[0])
        pos["label"] = True
        neg = pd.read_parquet(input[1])
        neg["label"] = False
        V = pd.concat([pos, neg], ignore_index=True)

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


#rule omim_subset_trait:
#    input:
#        "results/dataset/omim_matched_{k}/test.parquet",
#    output:
#        "results/dataset/omim_matched_{k}/subset/{t}.parquet",
#    wildcard_constraints:
#        t="|".join(omim_traits),
#    run:
#        V = pd.read_parquet(input[0])
#        target_size = 1 + int(wildcards.k)
#        V = V[(~V.label) | (V.OMIM == f"MIM {wildcards.t}")]
#        match_group_size = V.match_group.value_counts() 
#        match_groups = match_group_size[match_group_size == target_size].index
#        V = V[V.match_group.isin(match_groups)]
#        V[COORDINATES].to_parquet(output[0], index=False)
#