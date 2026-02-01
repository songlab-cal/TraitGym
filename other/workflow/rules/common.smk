import bioframe as bf
import polars_bio as pb
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
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
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


subsets = ["all"] + config["consequence_subsets"]


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


rule dataset_subset_consequence:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/{c}.parquet",
    wildcard_constraints:
        c="|".join(
            [
                "non_coding_transcript_exon_variant",
                "3_prime_UTR_variant",
                "5_prime_UTR_variant",
                "tss_proximal",
            ]
        ),
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence == wildcards.c]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_splicing:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/splicing.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence.isin(config["splicing_consequences"])]
        V[COORDINATES].to_parquet(output[0], index=False)


rule dataset_subset_nonexonic_distal:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/dataset/{dataset}/subset/nonexonic_distal.parquet",
    run:
        V = pd.read_parquet(input[0])
        V = V[V.consequence.isin(NON_EXONIC_FULL)]
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


rule matched_feature_performance:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        "results/matched_feature_performance/{dataset}.parquet",
    run:
        V = pl.read_parquet(input[0])

        # Feature signs: negative means lower value = higher score
        features = {
            "AF": -1,  # Lower AF (rarer) = more likely pathogenic
            "tss_dist": -1,  # Closer to TSS = more likely functional
            "exon_dist": -1,  # Closer to exon = more likely functional
        }

        rows = []
        # Baseline: proportion of positives
        baseline = V["label"].mean()
        rows.append({"feature": "baseline", "sign": 0, "AUPRC": baseline})

        # Each feature
        for feature, sign in features.items():
            auprc = average_precision_score(V["label"], V[feature] * sign)
            rows.append({"feature": feature, "sign": sign, "AUPRC": auprc})

        pl.DataFrame(rows).write_parquet(output[0])
