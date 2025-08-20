from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
import argparse
import concurrent.futures
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def run_vep(
    V,
    num_workers=0,
):
    model = dna_client.create(os.environ.get("ALPHA_GENOME_API_KEY"))
    metadata = model.output_metadata()
    sequence_length = '1MB'
    sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{sequence_length}'
    ]
    organism = dna_client.Organism.HOMO_SAPIENS

    tracks = [
        'ATAC',
        'DNASE',
        'CHIP_TF',
        'CHIP_HISTONE',
        'CAGE',
        'PROCAP',
        'RNA_SEQ',
    ]
    scorers = [
        variant_scorers.CenterMaskScorer(
            requested_output=getattr(dna_client.OutputType, track),
            width=None,
            aggregation_type=variant_scorers.AggregationType.L2_DIFF_LOG1P,
        )
        for track in tracks
    ]
    reverse_map = dict(zip(map(str, scorers), tracks))

    def run_vep_variant(v):
        variant = genome.Variant(
            chromosome=v.chrom,
            position=v.pos,
            reference_bases=v.ref,
            alternate_bases=v.alt,
        )
        interval = variant.reference_interval.resize(sequence_length)
        # need to put strand = "+" to fwd, since by default is "."
        interval_fwd = interval.copy()
        interval_fwd.strand = "+"
        interval_rev = interval.copy()
        interval_rev.strand = "-"
  
        def my_score_interval(interval):
            variant_scores = model.score_variant(
                interval=interval,
                variant=variant,
                organism=organism,
                variant_scorers=scorers,
            )
            return variant_scorers.tidy_scores([variant_scores])

        res_fwd = my_score_interval(interval_fwd)
        res_rev = my_score_interval(interval_rev)
        assert (res_fwd.index == res_rev.index).all() and (res_fwd.columns == res_rev.columns).all()
        res = res_fwd
        res.raw_score = (res.raw_score + res_rev.raw_score) / 2
        res["variant_scorer"] = res.variant_scorer.map(reverse_map)
        res["track"] = res["variant_scorer"] + "-" + res.groupby("variant_scorer").cumcount().astype(str)
        res = res.set_index("track")[["raw_score"]].T
        return res

    rows_iterable = V.itertuples(index=False)

    # simple version
    #res = list(tqdm(map(run_vep_variant, rows_iterable), total=len(V)))

    # parallel version (watch out for API limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        res = list(tqdm(executor.map(run_vep_variant, rows_iterable), total=len(V)))

    res = pd.concat(res, axis=0, ignore_index=True)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variant effect prediction"
    )
    parser.add_argument(
        "variants_path",
        type=str,
        help="Variants path. Needs the following columns: chrom,pos,ref,alt. pos should be 1-based",
    )
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--num_workers", type=int, default=0,
    )
    args = parser.parse_args()

    V = pd.read_parquet(args.variants_path, columns=["chrom", "pos", "ref", "alt"])
    V.chrom = "chr" + V.chrom

    pred = run_vep(V, num_workers=args.num_workers)
    
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pred.to_parquet(args.output_path, index=False)
