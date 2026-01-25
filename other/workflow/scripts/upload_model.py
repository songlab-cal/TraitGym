from huggingface_hub import HfApi
import os.path
import sys


model = sys.argv[1]


api = HfApi()
repo_id = "songlab/TraitGym"


def upload_file(path):
    if os.path.exists(path):
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
        )


datasets = [
    "mendelian_traits_matched_9",
    "complex_traits_matched_9",
]

metrics = [
    "AUPRC",
    "AUPRC_by_chrom",
    "AUROC_by_chrom",
    "AUPRC_by_chrom_weighted_average",
    "AUROC_by_chrom_weighted_average",
]

features = [
    "",
    "_LLR",
    "_absLLR",
    "_Embeddings",
    "_InnerProducts",
]

preds = [
    "_LLR.minus.score",
    "_absLLR.plus.score",
    "_Embeddings.plus.cosine_distance",
    "_Embeddings.minus.inner_product",
    "_Embeddings.plus.euclidean_distance",
    ".LogisticRegression.chrom",
    ".LogisticRegression.chrom.subset_from_all",
    ".plus.score",
]

subsets = [
    "all",
    "no_cadd_overlap",
]

for dataset in datasets:
    for feature in features:
        upload_file(f"{dataset}/features/{model}{feature}.parquet")
    for subset in subsets:
        for pred in preds:
            upload_file(f"{dataset}/preds/{subset}/{model}{pred}.parquet")
            for metric in metrics:
                upload_file(f"{dataset}/{metric}/{subset}/{model}{pred}.csv")
