from alphagenome.models import dna_client
import argparse
from Bio.Seq import Seq
import concurrent.futures
from gpn.data import Genome
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def run_vep(
    variants,
    genome,
    num_workers=0,
):
    model = dna_client.create(os.environ.get("ALPHA_GENOME_API_KEY"))
    window_size = 1_048_576

    metadata = model.output_metadata()

    output_names = [
        'ATAC',
        'CONTACT_MAPS',
        'DNASE',
        'CHIP_TF',
        'CHIP_HISTONE',
        'CAGE',
        'PROCAP',
        'RNA_SEQ',
        'SPLICE_SITES',
        'SPLICE_SITE_USAGE',
        #'SPLICE_JUNCTIONS',
        #'POLYADENYLATION',
    ]
    n_tracks = {x: len(getattr(metadata, x.lower())) for x in output_names}

    requested_outputs = [getattr(dna_client.OutputType, x) for x in output_names]

    def predict(seq):
        return model.predict_sequence(
            seq, requested_outputs=requested_outputs, ontology_terms=None,
        )

    def get_pred(output, output_name):
        x = getattr(output, output_name.lower()).values
        if output_name != "CONTACT_MAPS":
            x = np.log2(1+x)
        x = x.reshape(-1, x.shape[-1])
        return x

    def compute_score(output_ref, output_alt):
        res = []
        for output_name in output_names:
            pred_ref = get_pred(output_ref, output_name)
            pred_alt = get_pred(output_alt, output_name)
            score = np.linalg.norm(pred_alt-pred_ref, axis=0)
            res.append(score)
        res = np.concatenate(res)
        return res
        
    def run_vep_variant(v):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with Genome
        chrom = v.chrom
        pos = v.pos - 1
        start = pos - window_size // 2
        end = pos + window_size // 2
        print("before get seq")
        seq_fwd, seq_rev = genome.get_seq_fwd_rev(chrom, start, end)
        print("after get seq")
        ref_fwd = v.ref
        alt_fwd = v.alt
        ref_rev = str(Seq(ref_fwd).reverse_complement()) 
        alt_rev = str(Seq(alt_fwd).reverse_complement())
        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

        def process_strand(seq, pos, ref, alt):
            assert len(seq) == window_size
            seq = seq.upper()
            assert seq[pos] == ref, f"{seq[pos]}, {ref}"
            seq_ref = seq
            seq_alt = list(seq)
            seq_alt[pos] = alt
            seq_alt = "".join(seq_alt)
            print("before predict ref")
            pred_ref = predict(seq_ref)
            print("before predict alt")
            pred_alt = predict(seq_alt)
            print("before compute score")
            return compute_score(pred_ref, pred_alt)

        fwd = process_strand(seq_fwd, pos_fwd, ref_fwd, alt_fwd)
        rev = process_strand(seq_rev, pos_rev, ref_rev, alt_rev)
        return (fwd + rev) / 2


    rows_iterable = V.itertuples(index=False)
    
    #with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #    res = list(tqdm(executor.map(run_vep_variant, rows_iterable), total=len(V)))

    res = list(tqdm(map(run_vep_variant, rows_iterable), total=len(V)))
    
    res = np.vstack(res)
    print(res)
    print(res.shape)
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
    parser.add_argument(
        "genome_path",
        type=str,
        help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--num_workers", type=int, default=0,
    )
    args = parser.parse_args()

    #V = pd.read_parquet(args.variants_path, columns=["chrom", "pos", "ref", "alt"])
    V = pd.read_parquet(args.variants_path, columns=["chrom", "pos", "ref", "alt", "label"])
    V = V.groupby("label").sample(n=20, random_state=42).reset_index()

    genome = Genome(args.genome_path)

    pred = run_vep(V, genome, num_workers=args.num_workers)
    
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pred.to_parquet(args.output_path, index=False)
