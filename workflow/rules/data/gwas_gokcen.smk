gwas_gokcen_metadata = pd.read_csv(
    "results/gwas_gokcen/raw/disease_list.txt", delim_whitespace=True, header=None,
    names=["trait", "source", "name"]
)
gwas_gokcen_metadata["path"] = (
    gwas_gokcen_metadata.trait + "-" + gwas_gokcen_metadata.source
)
gwas_gokcen_metadata.set_index("trait", inplace=True)
eval_traits = [
    trait for trait in gwas_gokcen_metadata.index
    # < 5 non-exonic variants with PIP > 0.9 
    if trait not in [
        "SWB",
        "SCZ",
        "MDD",
        "BP",
        "SLE",
        "RA",
        "NRT",
        "Celiac",
        "Insomnia",
        "DHD",	
        "Celiac",
        "Insomnia",
        "ADHD",
    ]
]
print(f"{len(eval_traits)=}")

traits_high_n = [
    "Height",
    "BMD-HT",
    "PLT",
    "MCH",
    "HDL",
    "RDW",
    "RBC",
    "TG",
    "IBD",
    "CD",
    "HbA1c",
    "TC",
    "WHR",
    "BMI",
    "LDL",
    "HTN",
    "Glucose",
]

def get_input_gwas_gokcen_process(wildcards):
    path = gwas_gokcen_metadata.loc[wildcards.trait, "path"]
    return f"results/gwas_gokcen/raw/{path}.susie.gwfinemap.b38.gz"


rule gwas_gokcen_process:
    input:
        get_input_gwas_gokcen_process,
        "results/genome.fa.gz",
    output:
        "results/gwas_gokcen/processed/{trait}.parquet",
    run:
        V = (
            pl.read_csv(
                input[0], separator="\t",
                columns=["CHR", "BP", "A1", "A2", "MAF", "PIP", "P"],
            )
            # Filter out PIP > 0.9 but not genome-wide significant
            .filter(~((pl.col("PIP") > 0.9) & (pl.col("P") > 5e-8))).drop("P")
            .rename({"CHR": "chrom", "BP": "pos", "A1": "ref", "A2": "alt"})
            .to_pandas()
        )
        V.chrom = V.chrom.str.replace("chr", "")
        V = filter_chroms(V)
        V = filter_snp(V)
        genome = Genome(input[1])
        V = check_ref_alt(V, genome)
        V.to_parquet(output[0], index=False)


rule gwas_gokcen_merge_coords:
    input:
        expand(
            "results/gwas_gokcen/processed/{trait}.parquet",
            trait=gwas_gokcen_metadata.index
        ),
    output:
        "results/gwas_gokcen/coords.parquet",
    run:
        V = (
            pl.concat([pl.read_parquet(f, columns=COORDINATES) for f in input])
            .unique()
            .to_pandas()
        )
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_gokcen_trait_filt:
    input:
        "results/gwas_gokcen/coords.annot_with_cre.parquet",
        "results/gwas_gokcen/processed/{trait}.parquet",
    output:
        "results/gwas_gokcen/filt/{trait}.parquet",
    wildcard_constraints:
        trait="|".join(gwas_gokcen_metadata.index),
    run:
        annot = pd.read_parquet(input[0])
        V = (
            pl.read_parquet(input[1])
            .with_columns(
                pl.when(pl.col("PIP") > 0.9).then(True)
                .when(pl.col("PIP") < 0.01).then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
            .to_pandas()
        )
        V = V.merge(annot, on=COORDINATES)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_gokcen_merged_filt:
    input:
        "results/gwas_gokcen/coords.annot_with_cre.parquet",
        expand(
            "results/gwas_gokcen/processed/{trait}.parquet",
            trait=gwas_gokcen_metadata.index
        ),
    output:
        "results/gwas_gokcen/filt/merged.parquet",
    run:
        annot = pd.read_parquet(input[0])
        V = (
            pl.concat([
                pl.read_parquet(path).with_columns(
                    pl.when(pl.col("PIP") > 0.9)
                    .then(pl.lit(trait))
                    .otherwise(pl.lit(None))
                    .alias("trait")
                )
                for trait, path in tqdm(zip(gwas_gokcen_metadata.index, input[1:]))
            ])
            .group_by(COORDINATES)
            .agg(pl.max("PIP"), pl.mean("MAF"), pl.col("trait").drop_nulls().unique())
            .with_columns(
                pl.when(pl.col("PIP") > 0.9).then(True)
                .when(pl.col("PIP") < 0.01).then(False)
                .otherwise(None)
                .alias("label")
            )
            .drop_nulls()
            .with_columns(pl.col("trait").list.sort().list.join(","))
            .to_pandas()
        )
        V = V.merge(annot, on=COORDINATES)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


#rule gwas_gokcen_match:
#    input:
#        "results/gwas_gokcen/filt/{anything}.parquet",
#        "results/tss.parquet",
#    output:
#        "results/dataset/gwas_gokcen_{anything}_matched/test.parquet",
#    run:
#        V = pd.read_parquet(input[0])
#
#        V["start"] = V.pos - 1
#        V["end"] = V.pos
#
#        tss = pd.read_parquet(input[1], columns=["chrom", "start", "end"])
#
#        V = bf.closest(V, tss).rename(columns={
#            "distance": "tss_dist"
#        }).drop(columns=["start", "end", "chrom_", "start_", "end_"])
#
#        base_match_features = ["MAF"]
#
#        consequences = V[V.label].consequence.unique()
#        V_cs = []
#        for c in consequences:
#            print(c)
#            V_c = V[V.consequence == c].copy()
#            if c in NON_EXONIC + cre_classes + cre_flank_classes:
#                match_features = base_match_features + ["tss_dist"]
#            else:
#                match_features = base_match_features
#            for f in match_features:
#                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(V_c[f].values.reshape(-1, 1))
#            print(V_c.label.value_counts())
#            V_c = match_columns(V_c, "label", [f"{f}_scaled" for f in match_features])
#            V_c["match_group"] = c + "_" + V_c.match_group.astype(str)
#            print(V_c.label.value_counts())
#            print(V_c.groupby("label")[match_features].median())
#            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
#            V_cs.append(V_c)
#        V = pd.concat(V_cs, ignore_index=True)
#        V = sort_variants(V)
#        print(V)
#        V.to_parquet(output[0], index=False)


rule gwas_gokcen_match_k:
    input:
        "results/gwas_gokcen/filt/{anything}.parquet",
        "results/tss.parquet",
    output:
        "results/dataset/gwas_gokcen_{anything}_matched_{k,\d+}/test.parquet",
    run:
        k = int(wildcards.k)
        V = pd.read_parquet(input[0])

        V["start"] = V.pos - 1
        V["end"] = V.pos

        tss = pd.read_parquet(input[1], columns=["chrom", "start", "end"])

        V = bf.closest(V, tss).rename(columns={
            "distance": "tss_dist"
        }).drop(columns=["start", "end", "chrom_", "start_", "end_"])

        base_match_features = ["MAF"]

        consequences = V[V.label].consequence.unique()
        V_cs = []
        for c in consequences:
            print(c)
            V_c = V[V.consequence == c].copy()
            if c in NON_EXONIC + cre_classes + cre_flank_classes:
                match_features = base_match_features + ["tss_dist"]
            else:
                match_features = base_match_features
            for f in match_features:
                V_c[f"{f}_scaled"] = RobustScaler().fit_transform(V_c[f].values.reshape(-1, 1))
            print(V_c.label.value_counts())
            V_c = match_columns_k(V_c, "label", [f"{f}_scaled" for f in match_features], k)
            print(V_c.label.value_counts())
            print(V_c.groupby("label")[match_features].median())
            V_c.drop(columns=[f"{f}_scaled" for f in match_features], inplace=True)
            V_cs.append(V_c)
        V = pd.concat(V_cs, ignore_index=True)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_gokcen_filt_nonexonic:
    input:
        "results/dataset/gwas_gokcen_{anything}/test.parquet",
    output:
        "results/dataset/gwas_gokcen_{anything}_nonexonic/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence.isin(NON_EXONIC + cre_classes + cre_flank_classes)]
        print(V)
        V.to_parquet(output[0], index=False)


rule gwas_gokcen_filt_cre:
    input:
        "results/dataset/gwas_gokcen_{anything}/test.parquet",
    output:
        "results/dataset/gwas_gokcen_{anything}_cre/test.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence.isin(cre_classes + cre_flank_classes)]
        print(V)
        V.to_parquet(output[0], index=False)
