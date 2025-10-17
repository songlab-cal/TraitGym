from Bio.Seq import Seq
from evo2 import Evo2
from gpn.data import Genome
import pandas as pd
import torch
import sys


model_name = sys.argv[1]
genome_path = sys.argv[2]
chrom = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])
output_path = sys.argv[6]

genome = Genome(genome_path, subset_chroms=[chrom])

model = Evo2(model_name)

nucleotides = list("ACGT")
nucleotide_indices = model.tokenizer.tokenize("ACGT")

# let's average forward and rev, running 2 forward passes
# it's an approximation, could also run V * L forward passes
# where V=4 is the vocab size and L the sequence length

seq_fwd = genome.get_seq(chrom, start, end).upper()
seq_rev = str(Seq(seq_fwd).reverse_complement())


def get_logits(seq):
    input_ids = (
        torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        )
        .unsqueeze(0)
        .to("cuda:0")
    )

    with torch.inference_mode():
        (logits, _), _ = model(input_ids)
    logits = logits.float().cpu().numpy()

    logits = logits[0]
    logits = logits[:-2]
    logits = logits[:, nucleotide_indices]
    return logits


logits_fwd = get_logits(seq_fwd)
logits_rev = get_logits(seq_rev)[::-1, ::-1]  # reverse complement
logits = (logits_fwd + logits_rev) / 2

logits = pd.DataFrame(logits, columns=nucleotides)
logits["chrom"] = chrom
logits["pos"] = [x + 1 for x in range(start + 1, end - 1)]  # 1-based
logits["ref"] = list(seq_fwd[1:-1])
print(logits)
logits.to_parquet(output_path, index=False)
