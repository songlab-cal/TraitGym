rule eqtl_download:
    output:
        temp("results/eqtl/raw/Public_GTEx_finemapping.zip"),
        "results/eqtl/raw/GTEx_49tissues_release1.tsv.bgz",
    params:
        directory("results/eqtl/raw"),
    shell:
        """
        mkdir -p {params} &&
        wget -O {output[0]} "https://www.dropbox.com/scl/fo/bjp6o8hgixt5occ6ggq2o/ADTTMnyH-4rzDFS6g7oIcEE?rlkey=7558jp42yvmyjhgmcilbhlnu7&e=3&dl=1" &&
        cd {params} &&
        unzip {output[0]}
        """


# (base) shelob:~/projects/functionality-prediction/results/eqtl/raw$ zcat GTEx_49tissues_release1.tsv.bgz | head
# chromosome	start	end	variant	variant_hg38	allele1	allele2	minor_allele	cohort	method	tissue	gene	maf	beta_marginal	se_marginal	z	pip	cs_id	beta_posterior	sd_posterior
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Adipose_Subcutaneous	ENSG00000177757.2	0.0154905	0.322529	0.175853	1.83408	0.000297217	1	0.000121241	0.00791014
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Adipose_Subcutaneous	ENSG00000227232.5	0.0154905	0.243701	0.201673	1.2084	0.000297567	1	7.49224e-05	0.00564179
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Adipose_Subcutaneous	ENSG00000228463.9	0.0154905	0.330344	0.206138	1.60254	0.000204562	1	6.49145e-05	0.00523188
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Breast_Mammary_Tissue	ENSG00000227232.5	0.010101	-0.14507	0.333127	-0.43548	0.000127029	2	-6.93774e-06	0.00348231
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Cells_Cultured_fibroblasts	ENSG00000227232.5	0.0175983	0.168169	0.215392	0.780758	0.000140408	2	1.7601e-05	0.00291993
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Cells_Cultured_fibroblasts	ENSG00000237491.8	0.0175983	0.283539	0.177392	1.59838	0.000600595	2	0.000300969	0.0131954
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Colon_Transverse	ENSG00000227232.5	0.0108696	0.111084	0.289004	0.384368	0.000117537	2	7.71384e-06	0.00336039
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Lung	ENSG00000230092.7	0.0165049	0.372518	0.191018	1.95017	0.000340534	1	0.000147594	0.00892941
# chr1	13549	13550	chr1_13550_G_A	chr1_13550_G_A_b38	G	A	A	GTEx	FINEMAP	Lung	ENSG00000269981.1	0.0165049	0.106147	0.153807	0.690131	0.000104594	1	1.51622e-05	0.00264947


rule eqtl_process:
    input:
        "results/eqtl/raw/GTEx_49tissues_release1.tsv.bgz",
    output:
        "results/eqtl/processed.parquet",
    run:
        V = (
            pl.read_csv(
                input[0],
                separator="\t",
                columns=["variant_hg38", "method", "tissue", "gene", "maf", "pip", "z"],
            )
            .with_columns(
                pl.col("variant_hg38")
                .str.split_exact("_", 4)
                .struct.rename_fields(COORDINATES + ["build"])
            )
            .with_columns(
                pl.col("variant_hg38").struct.field("chrom").str.replace("chr", ""),
                pl.col("variant_hg38").struct.field("pos").cast(int),
                pl.col("variant_hg38").struct.field("ref"),
                pl.col("variant_hg38").struct.field("alt"),
            )
            .drop("variant_hg38")
        )
        print(V)
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
            ).select(
                ["chrom", "pos", "ref", "alt", "tissue", "gene", "method", "pip", "maf"]
            )
        )
        print(V)
        V = (
            V.group_by(["chrom", "pos", "ref", "alt", "tissue", "gene"])
            .agg(
                pl.mean("pip"),
                (pl.max("pip") - pl.min("pip")).alias("pip_diff"),
                pl.count().alias("pip_n"),
                pl.mean("maf"),
            )
            .filter(pl.col("pip_n") == 2, pl.col("pip_diff") < 0.05)
            .drop("pip_n", "pip_diff")
        )
        print(V)
        V = (
            V.with_columns(
                pl.when(pl.col("pip") > 0.9)
                .then(pl.col("tissue"))
                .otherwise(pl.lit(None))
                .alias("tissue")
            )
            .group_by(COORDINATES)
            .agg(pl.max("pip"), pl.col("tissue").drop_nulls().unique(), pl.mean("maf"))
            .with_columns(pl.col("tissue").list.sort().list.join(","))
            .to_pandas()
        )
        print(V)
        V = filter_snp(V)
        print(V.shape)
        V = sort_variants(V)
        print(V)
        V.to_parquet(output[0], index=False)


rule eqtl_positive:
    input:
        "results/eqtl/processed.annot_with_cre.parquet",
    output:
        "results/eqtl/pos.parquet",
    run:
        V = pl.read_parquet(input[0]).filter(pl.col("pip") > 0.9).to_pandas()
        V = V.drop_duplicates(COORDINATES)
        V = V[V.consequence.isin(TARGET_CONSEQUENCES)]
        assert len(V) == len(V.drop_duplicates(COORDINATES))
        print(V)
        V[COORDINATES].to_parquet(output[0], index=False)
