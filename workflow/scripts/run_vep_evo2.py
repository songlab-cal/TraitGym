import argparse
from Bio.Seq import Seq
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from evo2 import Evo2
from evo2.scoring import logits_to_logprobs
from gpn.data import Genome, load_dataset_from_file_or_dir


def euclidean_distance(embed_ref, embed_alt):
    B = len(embed_ref)
    return F.pairwise_distance(embed_ref.reshape(B, -1), embed_alt.reshape(B, -1))


def euclidean_distances(embed_ref, embed_alt):
    return torch.linalg.norm(embed_ref - embed_alt, dim=1)


def inner_product(embed_ref, embed_alt):
    return (embed_ref * embed_alt).sum(dim=(1, 2))


def inner_products(embed_ref, embed_alt):
    return (embed_ref * embed_alt).sum(dim=1)


def cosine_distance(embed_ref, embed_alt):
    B = len(embed_ref)
    return 1 - F.cosine_similarity(
        embed_ref.reshape(B, -1), embed_alt.reshape(B, -1), dim=1
    )


def cosine_distances(embed_ref, embed_alt):
    return 1 - F.cosine_similarity(embed_ref, embed_alt, dim=1)


class VEPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_ll_and_embed(self, input_ids):
        layer_name = "norm"  # last layer
        (logits, _), embed = self.model(
            input_ids, return_embeddings=True, layer_names=[layer_name]
        )
        ll = logits_to_logprobs(logits, input_ids).float().mean(dim=-1)
        embed = embed[layer_name]
        return ll, embed

    def get_scores(self, input_ids_ref, input_ids_alt):
        ll_ref, embed_ref = self.get_ll_and_embed(input_ids_ref)
        ll_alt, embed_alt = self.get_ll_and_embed(input_ids_alt)
        llr = ll_alt - ll_ref
        return torch.cat(
            (
                torch.unsqueeze(llr, 1),
                torch.unsqueeze(euclidean_distance(embed_ref, embed_alt), 1),
                torch.unsqueeze(inner_product(embed_ref, embed_alt), 1),
                torch.unsqueeze(cosine_distance(embed_ref, embed_alt), 1),
                euclidean_distances(embed_ref, embed_alt),
                inner_products(embed_ref, embed_alt),
                cosine_distances(embed_ref, embed_alt),
            ),
            dim=1,
        )

    def forward(
        self,
        input_ids_ref_fwd=None,
        input_ids_alt_fwd=None,
        input_ids_ref_rev=None,
        input_ids_alt_rev=None,
    ):
        fwd = self.get_scores(input_ids_ref_fwd, input_ids_alt_fwd)
        rev = self.get_scores(input_ids_ref_rev, input_ids_alt_rev)
        return (fwd + rev) / 2


def run_vep(
    variants,
    genome,
    tokenizer,
    model,
    window_size,
    per_device_batch_size=8,
    dataloader_num_workers=0,
):
    def tokenize(seqs):
        return tokenizer.tokenize_batch(seqs)

    def get_tokenized_seq(vs):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with Genome
        chrom = np.array(vs["chrom"])
        n = len(chrom)
        pos = np.array(vs["pos"]) - 1
        start = pos - window_size // 2
        end = pos + window_size // 2
        seq_fwd, seq_rev = zip(
            *(genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n))
        )
        seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd], dtype="object")
        seq_rev = np.array([list(seq.upper()) for seq in seq_rev], dtype="object")
        assert seq_fwd.shape[1] == window_size
        assert seq_rev.shape[1] == window_size
        ref_fwd = np.array(vs["ref"])
        alt_fwd = np.array(vs["alt"])
        ref_rev = np.array([str(Seq(x).reverse_complement()) for x in ref_fwd])
        alt_rev = np.array([str(Seq(x).reverse_complement()) for x in alt_fwd])
        pos_fwd = window_size // 2
        pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

        def prepare_output(seq, pos, ref, alt):
            assert (seq[:, pos] == ref).all(), f"{seq[:, pos]}, {ref}"
            seq_ref = seq
            seq_alt = seq.copy()
            seq_alt[:, pos] = alt
            return (
                np.array(tokenize(["".join(x) for x in seq_ref]), dtype=np.int64),
                np.array(tokenize(["".join(x) for x in seq_alt]), dtype=np.int64),
            )

        res = {}
        res["input_ids_ref_fwd"], res["input_ids_alt_fwd"] = prepare_output(
            seq_fwd, pos_fwd, ref_fwd, alt_fwd
        )
        res["input_ids_ref_rev"], res["input_ids_alt_rev"] = prepare_output(
            seq_rev, pos_rev, ref_rev, alt_rev
        )
        return res

    variants.set_transform(get_tokenized_seq)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        # torch_compile=True,
        # bf16=True, bf16_full_eval=True,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=variants).predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot variant effect prediction with AutoModelForMaskedLM"
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
    parser.add_argument("window_size", type=int, help="Genomic window size")
    parser.add_argument("model_path", help="Model path (local or on HF hub)", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Tokenizer path (optional, else will use model_path)",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--is_file",
        action="store_true",
        help="VARIANTS_PATH is a file, not directory",
    )
    args = parser.parse_args()

    variants = load_dataset_from_file_or_dir(
        args.variants_path,
        split=args.split,
        is_file=args.is_file,
    )
    subset_chroms = np.unique(variants["chrom"])
    genome = Genome(args.genome_path, subset_chroms=subset_chroms)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    evo2 = Evo2(args.model_path)
    tokenizer = evo2.tokenizer
    model = VEPWrapper(evo2)
    pred = run_vep(
        variants,
        genome,
        tokenizer,
        model,
        args.window_size,
        per_device_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    D = (pred.shape[1] - 4) // 3
    cols = (
        ["llr", "euclidean_distance", "inner_product", "cosine_distance"]
        + [f"euclidean_distance_{i}" for i in range(D)]
        + [f"inner_product_{i}" for i in range(D)]
        + [f"cosine_distance_{i}" for i in range(D)]
    )
    directory = os.path.dirname(args.output_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
    pd.DataFrame(pred, columns=cols).to_parquet(args.output_path, index=False)
