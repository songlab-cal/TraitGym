from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
import argparse
import os
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variant effect prediction"
    )
    parser.add_argument("input_path", help="Input path (parquet)", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    args = parser.parse_args()

    dna_model = dna_client.create(os.environ.get("ALPHA_GENOME_API_KEY"))

    V = pd.read_parquet(args.input_path, columns=["chrom", "pos", "ref", "alt"])
    V.chrom = "chr" + V.chrom

    reverse_map = {str(v): k for k, v in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items()}
    scorers = [scorer for name, scorer in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items() if "ACTIVE" not in name]
    sequence_length = '1MB'
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{sequence_length}'
    ]
    organism = dna_client.Organism.HOMO_SAPIENS

    all_res = []
    for i, v in tqdm(V.iterrows(), total=len(V)):
        variant = genome.Variant(
            chromosome=v.chrom,
            position=v.pos,
            reference_bases=v.ref,
            alternate_bases=v.alt,
        )
        interval = variant.reference_interval.resize(sequence_length)
        variant_scores = dna_model.score_variant(
            interval=interval,
            variant=variant,
            organism=organism,
            variant_scorers=scorers,
        )
        res = variant_scorers.tidy_scores([variant_scores])
        res["variant_scorer"] = res.variant_scorer.map(reverse_map)
        res["abs_raw_score"] = res.raw_score.abs()
        res["abs_quantile_score"] = res.quantile_score.abs()
        res = res.groupby("variant_scorer")[["abs_raw_score", "abs_quantile_score"]].max()
        res.rename(columns={"abs_raw_score": "max_abs_raw_score", "abs_quantile_score": "max_abs_quantile_score"}, inplace=True)
        res.loc["all"] = res.max()
        res = res.unstack()
        res.index = [f'{level0}_{level1}' for level0, level1 in res.index]
        res = res.to_frame().T
        all_res.append(res)
    res = pd.concat(all_res, axis=0, ignore_index=True).fillna(0)

    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    res.to_parquet(args.output_path, index=False)
