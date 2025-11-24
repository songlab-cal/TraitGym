import bioframe as bf
from cyvcf2 import VCF
from datasets import load_dataset
from gpn.data import Genome, load_table, load_dataset_from_file_or_dir
from gnomad_db.database import gnomAD_DB
from liftover import get_lifter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
import polars.selectors as cs
from scipy.spatial.distance import cdist
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.special import softmax
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import yaml


COORDINATES = ["chrom", "pos", "ref", "alt"]
NUCLEOTIDES = list("ACGT")
ODD_EVEN_CHROMS = [
    [str(i) for i in range(1, 23, 2)] + ["X"],
    [str(i) for i in range(2, 23, 2)] + ["Y"],
]
CHROMS = [str(i) for i in range(1, 23)]
NON_EXONIC = [
    "intergenic_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
]

# order is important since it will determine which flank gets priority
# following order in https://screen.wenglab.org/about
# roughly from most evidence of functional importance to least
CRE_CLASSES = [
    "PLS",
    "pELS",
    "dELS",
    "CA-H3K4me3",
    "CA-CTCF",
    "CA-TF",
    "CA",
    "TF",
]
CRE_FLANK_CLASSES = [f"{c}_flank" for c in CRE_CLASSES]
NON_EXONIC_FULL = NON_EXONIC + CRE_CLASSES + CRE_FLANK_CLASSES
other_consequences = [
    "missense_variant",
    "non_coding_transcript_exon_variant",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
]

TARGET_CONSEQUENCES = NON_EXONIC_FULL + [
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "non_coding_transcript_exon_variant",
]

select_gwas_traits = (
    pd.read_csv("config/gwas/independent_traits_filtered.csv", header=None)
    .values.ravel()
    .tolist()
)
select_gwas_traits_n30 = (
    pd.read_csv("config/gwas/independent_traits_filtered_n30.csv", header=None)
    .values.ravel()
    .tolist()
)
select_omim_traits = (
    pd.read_csv("config/omim/filtered_traits.txt", header=None, dtype=str)
    .values.ravel()
    .tolist()
)

tissues = pd.read_csv("config/gtex_tissues.txt", header=None).values.ravel()


def add_tss(V: pl.DataFrame, tss: pl.DataFrame) -> pl.DataFrame:
    """Add tss_dist column with distance to nearest TSS.

    Also overrides consequence to "tss_proximal" for non-exonic variants
    within tss_proximal_dist of a TSS.
    """
    V_pd = V.to_pandas()
    tss_pd = tss.to_pandas()
    V_pd["start"] = V_pd.pos - 1
    V_pd["end"] = V_pd.pos
    V_pd = (
        bf.closest(V_pd, tss_pd)
        .rename(columns={"distance": "tss_dist", "gene_id_": "tss_closest_gene_id"})
        .drop(columns=["start", "end", "chrom_", "start_", "end_"])
    )
    tss_proximal_dist = config["tss_proximal_dist"]
    mask = V_pd.original_consequence.isin(NON_EXONIC) & (
        V_pd.tss_dist <= tss_proximal_dist
    )
    V_pd.loc[mask, "consequence"] = "tss_proximal"
    return pl.from_pandas(V_pd)


def add_exon(V: pl.DataFrame, exon: pl.DataFrame) -> pl.DataFrame:
    """Add exon_dist column with distance to nearest exon.

    Also overrides consequence to "exon_proximal" for intron_variant
    within exon_proximal_dist of an exon.
    """
    V_pd = V.to_pandas()
    exon_pd = exon.to_pandas()
    V_pd["start"] = V_pd.pos - 1
    V_pd["end"] = V_pd.pos
    V_pd = (
        bf.closest(V_pd, exon_pd)
        .rename(columns={"distance": "exon_dist", "gene_id_": "exon_closest_gene_id"})
        .drop(columns=["start", "end", "chrom_", "start_", "end_"])
    )
    exon_proximal_dist = config["exon_proximal_dist"]
    mask = (V_pd.original_consequence == "intron_variant") & (
        V_pd.exon_dist <= exon_proximal_dist
    )
    V_pd.loc[mask, "consequence"] = "exon_proximal"
    return pl.from_pandas(V_pd)


def add_cre(V: pl.DataFrame, cre: pl.DataFrame) -> pl.DataFrame:
    """Add CRE-based consequence annotations.

    For non-exonic variants, updates consequence to CRE class or CRE flank
    based on overlap with cis-regulatory elements.
    Requires original_consequence column to be present.
    """
    V_pd = V.to_pandas()
    cre_pd = cre.to_pandas()

    V_pd["start"] = V_pd.pos - 1
    V_pd["end"] = V_pd.pos

    # First assign flank classes (expanded by 500bp)
    # Process in reverse order so higher priority classes override
    for c in reversed(CRE_CLASSES):
        I = cre_pd[cre_pd.cre_class == c]
        I = bf.expand(I, pad=config["cre_flank_dist"])
        I = bf.merge(I).drop(columns="n_intervals")
        V_pd = bf.coverage(V_pd, I)
        mask = V_pd.original_consequence.isin(NON_EXONIC) & (V_pd.coverage > 0)
        V_pd.loc[mask, "consequence"] = f"{c}_flank"
        V_pd = V_pd.drop(columns=["coverage"])

    # Then assign main CRE classes (overriding flanks)
    for c in reversed(CRE_CLASSES):
        I = cre_pd[cre_pd.cre_class == c]
        V_pd = bf.coverage(V_pd, I)
        mask = V_pd.original_consequence.isin(NON_EXONIC) & (V_pd.coverage > 0)
        V_pd.loc[mask, "consequence"] = c
        V_pd = V_pd.drop(columns=["coverage"])

    V_pd = V_pd.drop(columns=["start", "end"])
    return pl.from_pandas(V_pd)


def filter_snp(V: pd.DataFrame) -> pd.DataFrame:
    return V[V.ref.isin(NUCLEOTIDES) & V.alt.isin(NUCLEOTIDES)]


def filter_chroms(V: pd.DataFrame) -> pd.DataFrame:
    return V[V.chrom.isin(CHROMS)]


def lift_hg19_to_hg38(V: pd.DataFrame) -> pd.DataFrame:
    converter = get_lifter("hg19", "hg38")

    def get_new_pos(v):
        try:
            res = converter[v.chrom][v.pos]
            assert len(res) == 1
            chrom, pos, strand = res[0]
            assert chrom.replace("chr", "") == v.chrom
            return pos
        except:
            return -1

    V.pos = V.apply(get_new_pos, axis=1)
    return V


def sort_variants(V: pd.DataFrame) -> pd.DataFrame:
    V.chrom = pd.Categorical(V.chrom, categories=CHROMS, ordered=True)
    V = V.sort_values(COORDINATES)
    V.chrom = V.chrom.astype(str)
    return V


def check_ref(V, genome):
    V = V[V.apply(lambda v: v.ref == genome.get_nuc(v.chrom, v.pos).upper(), axis=1)]
    return V


def check_ref_alt(V: pd.DataFrame, genome: Genome) -> pd.DataFrame:
    V["ref_nuc"] = V.progress_apply(
        lambda v: genome.get_nuc(v.chrom, v.pos).upper(), axis=1
    )
    mask = V["ref"] != V["ref_nuc"]
    V.loc[mask, ["ref", "alt"]] = V.loc[mask, ["alt", "ref"]].values
    V = V[V["ref"] == V["ref_nuc"]]
    V.drop(columns=["ref_nuc"], inplace=True)
    return V


def match_features(
    pos: pl.DataFrame,
    neg: pl.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    k: int,
    scale: bool = True,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Match positive samples to k negative samples based on features.

    For each unique combination of categorical features, finds the k closest
    negative samples for each positive sample based on continuous features.

    Args:
        pos: Positive samples DataFrame.
        neg: Negative samples DataFrame.
        continuous_features: Columns to use for distance-based matching.
        categorical_features: Columns for exact matching (grouping).
        k: Number of negative samples to match per positive sample.
        scale: Whether to scale continuous features before matching.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with matched samples and match_group column.
    """
    # Convert to pandas for internal processing
    pos = pos.to_pandas()
    neg = neg.to_pandas()

    # Scale continuous features if requested
    if scale and len(continuous_features) > 0:
        scaler = RobustScaler()
        all_data = pd.concat([pos[continuous_features], neg[continuous_features]])
        scaler.fit(all_data)
        pos[continuous_features] = scaler.transform(pos[continuous_features])
        neg[continuous_features] = scaler.transform(neg[continuous_features])

    pos = pos.set_index(categorical_features)
    neg = neg.set_index(categorical_features)
    res_pos = []
    res_neg = []
    for x in tqdm(pos.index.drop_duplicates()):
        pos_x = pos.loc[[x]].reset_index()
        try:
            neg_x = neg.loc[[x]].reset_index()
        except KeyError:
            print(f"WARNING: no match for {x}")
            continue
        if len(pos_x) * k > len(neg_x):
            print("WARNING: subsampling positive set")
            pos_x = pos_x.sample(len(neg_x) // k, random_state=seed)
        if len(continuous_features) == 0:
            neg_x = neg_x.sample(len(pos_x) * k, random_state=seed)
        else:
            neg_x = _find_closest(pos_x, neg_x, continuous_features, k)
        res_pos.append(pos_x)
        res_neg.append(neg_x)
    res_pos = pd.concat(res_pos, ignore_index=True)
    res_pos["match_group"] = np.arange(len(res_pos))
    res_neg = pd.concat(res_neg, ignore_index=True)
    res_neg["match_group"] = np.repeat(res_pos.match_group.values, k)
    res = pd.concat([res_pos, res_neg], ignore_index=True)
    return pl.from_pandas(res)


def _find_closest(
    pos: pd.DataFrame, neg: pd.DataFrame, cols: list[str], k: int
) -> pd.DataFrame:
    """Find k closest negative samples for each positive sample."""
    D = cdist(pos[cols], neg[cols])
    closest = []
    for i in range(len(pos)):
        js = np.argsort(D[i])[:k].tolist()
        closest += js
        D[:, js] = np.inf  # ensure they cannot be picked up again
    return neg.iloc[closest]


rule download_genome:
    output:
        "results/genome.fa.gz",
    shell:
        "wget -O {output} {config[genome_url]}"


rule download_annotation:
    output:
        "results/annotation.gtf.gz",
    shell:
        "wget -O {output} {config[annotation_url]}"


rule get_tss:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/tss.parquet",
    run:
        annotation = load_table(input[0])
        tx = annotation.query('feature=="transcript"').copy()
        tx["gene_id"] = tx.attribute.str.extract(r'gene_id "([^;]*)";')
        tx["transcript_biotype"] = tx.attribute.str.extract(
            r'transcript_biotype "([^;]*)";'
        )
        tx = tx[tx.transcript_biotype == "protein_coding"]
        tss = tx.copy()
        tss[["start", "end"]] = tss.progress_apply(
            lambda w: (w.start, w.start + 1) if w.strand == "+" else (w.end - 1, w.end),
            axis=1,
            result_type="expand",
        )
        tss = tss[["chrom", "start", "end", "gene_id"]]
        print(tss)
        tss.to_parquet(output[0], index=False)


rule get_exon:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/exon.parquet",
    run:
        annotation = load_table(input[0])
        exon = annotation.query('feature=="exon"').copy()
        exon["gene_id"] = exon.attribute.str.extract(r'gene_id "([^;]*)";')
        exon["transcript_biotype"] = exon.attribute.str.extract(
            r'transcript_biotype "([^;]*)";'
        )
        exon = exon[exon.transcript_biotype == "protein_coding"]
        exon = exon[["chrom", "start", "end", "gene_id"]].drop_duplicates()
        exon = exon[exon.chrom.isin(CHROMS)]
        exon = exon.sort_values(["chrom", "start", "end"])
        print(exon)
        exon.to_parquet(output[0], index=False)


rule make_ensembl_vep_input:
    input:
        "{anything}.parquet",
    output:
        #temp("{anything}.ensembl_vep.input.tsv.gz"),
        "{anything}.ensembl_vep.input.tsv.gz",
    threads: workflow.cores
    run:
        df = pd.read_parquet(input[0])
        df["start"] = df.pos
        df["end"] = df.start
        df["allele"] = df.ref + "/" + df.alt
        df["strand"] = "+"
        df.to_csv(
            output[0],
            sep="\t",
            header=False,
            index=False,
            columns=["chrom", "start", "end", "allele", "strand"],
        )


# additional snakemake args (SCF):
# --sdm apptainer --apptainer-args "--bind /scratch/users/gbenegas"
# or in savio:
# --sdm apptainer --apptainer-args "--bind /global/scratch/projects/fc_songlab/gbenegas"
rule install_ensembl_vep_cache:
    output:
        directory("results/ensembl_vep_cache"),
    container:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        "INSTALL.pl -c {output} -a cf -s homo_sapiens -y GRCh38"


rule run_ensembl_vep:
    input:
        "{anything}.ensembl_vep.input.tsv.gz",
        "results/ensembl_vep_cache",
    output:
        #temp("{anything}.ensembl_vep.output.tsv.gz"),
        #temp("{anything}.ensembl_vep.output.tsv.gz_summary.html"),
        "{anything}.ensembl_vep.output.tsv.gz",
        "{anything}.ensembl_vep.output.tsv.gz_summary.html",
    container:
        "docker://ensemblorg/ensembl-vep:release_109.1"
    threads: workflow.cores
    shell:
        """
        vep -i {input[0]} -o {output} --fork {threads} --cache \
        --dir_cache {input[1]} --format ensembl \
        --most_severe --compress_output gzip --tab --distance 1000 --offline
        """


rule process_ensembl_vep:
    input:
        "{anything}.parquet",
        "{anything}.ensembl_vep.output.tsv.gz",
    output:
        "{anything}.annot.parquet",
    priority: 100
    run:
        V = pl.read_parquet(input[0])
        V2 = pl.read_csv(
            input[1],
            separator="\t",
            has_header=False,
            comment_prefix="#",
            new_columns=["variant", "consequence"],
            columns=[0, 6],
        )
        V2 = V2.with_columns(
            pl.col("variant").str.split("_").list.get(0).alias("chrom"),
            pl.col("variant").str.split("_").list.get(1).cast(pl.Int64).alias("pos"),
            pl.col("variant")
            .str.split("_")
            .list.get(2)
            .str.split("/")
            .list.get(0)
            .alias("ref"),
            pl.col("variant")
            .str.split("_")
            .list.get(2)
            .str.split("/")
            .list.get(1)
            .alias("alt"),
        ).drop("variant")
        V = V.join(V2, on=COORDINATES, how="left", maintain_order="left")
        V.write_parquet(output[0])


rule upload_features_to_hf:
    input:
        "results/features/{dataset}/{features}.parquet",
    output:
        touch("results/features/{dataset}/{features}.parquet.uploaded"),
    threads: workflow.cores
    run:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=input[0],
            path_in_repo=f"features/{wildcards.features}.parquet",
            repo_id=wildcards.dataset,
            repo_type="dataset",
        )


def predict(clf, X):
    return clf.predict_proba(X)[:, 1]


def train_predict(V_train, V_test, features, train_f):
    clf = train_f(V_train[features], V_train.label, V_train.chrom)
    return predict(clf, V_test[features])


def train_logistic_regression(X, y, groups):
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    missing_values=np.nan,
                    strategy="mean",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "linear",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    Cs = np.logspace(-8, 0, 10)
    param_grid = {
        "linear__C": Cs,
    }
    clf = GridSearchCV(
        pipeline,
        param_grid,
        scoring="average_precision",
        cv=GroupKFold(),
        n_jobs=-1,
    )
    clf.fit(X, y, groups=groups)
    print(f"{clf.best_params_=}")
    linear = clf.best_estimator_.named_steps["linear"]
    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "coef": linear.coef_[0],
        }
    ).sort_values("coef", ascending=False, key=abs)
    print(coef.head(10))
    return clf


classifier_map = {
    "LogisticRegression": train_logistic_regression,
    # "BestFeature": train_best_feature,
    # "RandomForest": train_random_forest,
    # "XGBoost": train_xgboost,
    # "PCALogisticRegression": train_pca_logistic_regression,
    # "FeatureSelectionLogisticRegression": train_feature_selection_logistic_regression,
    # "RidgeRegression": train_ridge_regression,
}


def format_number(num):
    """
    Converts a number into a more readable format, using K for thousands, M for millions, etc.
    Args:
    - num: The number to format.

    Returns:
    - A formatted string representing the number.
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


rule download_s_het:
    output:
        "results/s_het.xlsx",
    shell:
        "wget -O {output} https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-024-01820-9/MediaObjects/41588_2024_1820_MOESM4_ESM.xlsx"


rule process_s_het:
    input:
        "results/s_het.xlsx",
    output:
        "results/s_het.parquet",
    run:
        df = pl.read_excel(input[0], sheet_name="Supplementary Table 1").to_pandas()
        df = df[["ensg", "post_mean"]].rename(
            columns={"ensg": "gene_id", "post_mean": "s_het"}
        )
        df.to_parquet(output[0], index=False)


rule tss_s_het:
    input:
        "results/tss.parquet",
        "results/s_het.parquet",
    output:
        "results/tss_s_het.parquet",
    run:
        tss = pd.read_parquet(input[0])
        s_het = pd.read_parquet(input[1])
        tss = tss.merge(s_het, on="gene_id", how="inner")
        tss.to_parquet(output[0], index=False)


rule s_het_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/tss_s_het.parquet",
    output:
        "results/dataset/{dataset}/features/s_het.parquet",
    run:
        V = pd.read_parquet(input[0])
        original_order = V.pos.values
        tss = pd.read_parquet(input[1])
        V["start"] = V.pos - 1
        V["end"] = V.pos
        V = bf.closest(V, tss).rename(columns={"s_het_": "s_het"})
        V = sort_variants(V)
        new_order = V.pos.values
        assert np.all(original_order == new_order)
        V[["s_het"]].to_parquet(output[0], index=False)


rule abs_llr:
    input:
        "results/dataset/{dataset}/features/{model}_LLR.parquet",
    output:
        "results/dataset/{dataset}/features/{model}_absLLR.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df.abs()
        df.to_parquet(output[0], index=False)


rule inner_product:
    input:
        "results/dataset/{dataset}/features/{model}_InnerProducts.parquet",
    output:
        "results/dataset/{dataset}/features/{model}_InnerProduct.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = df.sum(axis=1).rename("score").to_frame()
        df.to_parquet(output[0], index=False)


rule euclidean_distance:
    input:
        "results/dataset/{dataset}/features/{model}_EuclideanDistances.parquet",
    output:
        "results/dataset/{dataset}/features/{model}_EuclideanDistance.parquet",
    run:
        df = pd.read_parquet(input[0])
        df = pd.DataFrame({"score": np.linalg.norm(df.values, axis=1)})
        df.to_parquet(output[0], index=False)


rule extract_inner_products:
    input:
        "results/dataset/{dataset}/features/{model}_Embeddings.parquet",
    output:
        "results/dataset/{dataset}/features/{model}_InnerProducts.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .select(cs.starts_with("inner_product_"))
            .write_parquet(output[0])
        )


rule dataset_subset_defined_alphamissense:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/features/AlphaMissense.parquet",
    output:
        "results/dataset/{dataset}/subset/defined_alphamissense.parquet",
    run:
        V = pd.concat([pd.read_parquet(input[0]), pd.read_parquet(input[1])], axis=1)
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        V = V[V.consequence == "missense_variant"]
        V = V.dropna(subset="score")
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V)
        V[COORDINATES].to_parquet(output[0], index=False)


def bootstrap_se(df, stat, n_bootstraps=1000):
    return pl.Series(
        [
            stat(df.sample(len(df), with_replacement=True, seed=i))
            for i in range(n_bootstraps)
        ]
    ).std()


def block_bootstrap_se(
    metric, df, y_true_col, y_pred_col, block_col, n_bootstraps=1000
):
    df = pl.DataFrame(df)
    all_blocks = df[block_col].unique().sort()
    df_blocks = {
        block: df.filter(pl.col(block_col) == block).select([y_pred_col, y_true_col])
        for block in all_blocks
    }
    bootstraps = []
    for i in range(n_bootstraps):
        boot_blocks = all_blocks.sample(len(all_blocks), with_replacement=True, seed=i)
        df_boot = pl.concat([df_blocks[block] for block in boot_blocks])
        bootstraps.append(metric(df_boot[y_true_col], df_boot[y_pred_col]))
    bootstraps = pl.Series(bootstraps)
    return bootstraps.std()


rule dataset_subset_all:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/all.parquet",
    run:
        V = pd.read_parquet(input[0])
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_non_missense:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/non_missense.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence != "missense_variant"]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_non_coding:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/non_coding.parquet",
    run:
        V = pd.read_parquet(input[0])
        # these are the ones that appear in our datasets
        exclude = [
            "missense_variant",
            "synonymous_variant",
            "stop_gained",
        ]
        V = V[~V.consequence.isin(exclude)]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_non_coding_v2:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/non_coding_v2.parquet",
    run:
        V = pd.read_parquet(input[0])
        include = (
            NON_EXONIC
            + cre_classes
            + cre_flank_classes
            + [
                "5_prime_UTR_variant",
                "3_prime_UTR_variant",
                "non_coding_transcript_exon_variant",
            ]
        )
        V = V[V.consequence.isin(include)]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_nonexonic:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/nonexonic.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence.isin(NON_EXONIC + cre_classes + cre_flank_classes)]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_consequence:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/{c}.parquet",
    wildcard_constraints:
        c="|".join(other_consequences),
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence == wildcards.c]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_proximal:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/proximal.parquet",
    run:
        V = pd.read_parquet(input[0])
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        V = V[(~V.label) | (V.tss_dist <= 1_000)]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_distal:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/distal.parquet",
    run:
        V = pd.read_parquet(input[0])
        target_size = len(V[V.match_group == V.match_group.iloc[0]])
        V = V[(~V.label) | (V.tss_dist > 1_000)]
        match_group_size = V.match_group.value_counts()
        match_groups = match_group_size[match_group_size == target_size].index
        V = V[V.match_group.isin(match_groups)]
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_no_cadd_overlap:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/cadd/train.parquet",
    output:
        "results/dataset/{dataset}/subset/no_cadd_overlap.parquet",
    run:
        V = pd.read_parquet(input[0])
        print(V)
        if "match_group" in V.columns:
            target_size = len(V[V.match_group == V.match_group.iloc[0]])
        cadd = pd.read_parquet(input[1])
        V = V.merge(cadd, on=COORDINATES, how="left")
        V = V[V.cadd_label.isna()]
        if "match_group" in V.columns:
            match_group_size = V.match_group.value_counts()
            match_groups = match_group_size[match_group_size == target_size].index
            V = V[V.match_group.isin(match_groups)]
        print(V)
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_eqtl_overlap:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/eqtl/pos.parquet",
    output:
        "results/dataset/{dataset}/subset/{do,yes|no}_eqtl_overlap.parquet",
    run:
        V = pd.read_parquet(input[0])
        print(V)
        if "match_group" in V.columns:
            target_size = len(V[V.match_group == V.match_group.iloc[0]])
        eqtl = pd.read_parquet(input[1])
        eqtl["eqtl"] = True
        V = V.merge(eqtl, on=COORDINATES, how="left")
        if wildcards.do == "yes":
            V = V[(~V.label) | (~V.eqtl.isna())]
        elif wildcards.do == "no":
            V = V[(~V.label) | (V.eqtl.isna())]
        if "match_group" in V.columns:
            match_group_size = V.match_group.value_counts()
            match_groups = match_group_size[match_group_size == target_size].index
            V = V[V.match_group.isin(match_groups)]
        print(V)
        print(V.label.value_counts())
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_intersect:
    input:
        "results/dataset/{dataset}/subset/{s1}.parquet",
        "results/dataset/{dataset}/subset/{s2}.parquet",
    output:
        "results/dataset/{dataset}/subset/{s1}_AND_{s2}.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .join(pl.read_parquet(input[1]), how="inner", on=COORDINATES)
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


# for Sei, experimental
rule dataset_to_vcf:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/vcf/{dataset}.vcf",
    run:
        V = pd.read_parquet(input[0])
        V.chrom = "chr" + V.chrom
        V["id"] = "."
        V[["chrom", "pos", "id", "ref", "alt"]].to_csv(
            output[0],
            sep="\t",
            index=False,
            header=False,
        )


rule dataset_to_vcf_gz:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/vcf/{dataset}.vcf.gz",
    run:
        V = pd.read_parquet(input[0])
        V["id"] = "."
        V[["chrom", "pos", "id", "ref", "alt"]].to_csv(
            output[0],
            sep="\t",
            index=False,
            header=False,
        )
