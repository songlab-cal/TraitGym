def generate_possible_snvs(seq: str, chrom: str) -> pl.DataFrame:
    """
    Generates a Polars DataFrame of all possible single nucleotide variants (SNVs)
    from a reference DNA sequence using the most current and efficient Polars API.

    This function ignores any positions in the reference sequence where the
    base is not one of 'A', 'C', 'G', or 'T'.

    Args:
        seq: The reference DNA sequence for a chromosome.
        chrom: The identifier of the chromosome (e.g., 'chr1', 'X').

    Returns:
        V: A Polars DataFrame containing all conceivable SNVs from the input
           chromosome sequence. The DataFrame includes the following columns:
           - 'chrom': The chromosome identifier.
           - 'pos': The 1-based position of the variant on the chromosome.
           - 'ref': The reference base at that position.
           - 'alt': The alternative (variant) base.
    """
    # Create a DataFrame from the input DNA sequence.
    df = pl.DataFrame({"ref": list(seq)})

    # Add a 'pos' column with the 1-based position using the modern API.
    df = df.with_row_index("pos", offset=1)

    # Define the mapping from each reference base to its possible alternatives.
    valid_bases = {"A", "C", "G", "T"}
    alt_map = {
        "A": ["C", "G", "T"],
        "C": ["A", "G", "T"],
        "G": ["A", "C", "T"],
        "T": ["A", "C", "G"],
    }

    # Filter the DataFrame to include only rows with valid DNA bases.
    df = df.filter(pl.col("ref").is_in(list(valid_bases)))

    # Use the recommended 'replace_strict' for fast, native mapping.
    # This is the efficient and correct replacement for this pattern.
    df = df.with_columns(pl.col("ref").replace_strict(alt_map).alias("alt"))

    # "Explode" the 'alt' column to give each SNV its own row.
    df = df.explode("alt")

    # Add the constant 'chrom' column.
    df = df.with_columns(pl.lit(chrom).alias("chrom"))

    # Select and reorder columns to the final format.
    V = df.select(["chrom", "pos", "ref", "alt"])

    return V


rule simulated_variants:
    input:
        "results/genome.fa.gz",
    output:
        "results/simulated/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        chrom = wildcards.chrom
        seq = Genome(input[0], subset_chroms=[chrom])._genome[chrom].upper()
        V = generate_possible_snvs(seq, chrom)
        print(V)
        V.write_parquet(output[0])


rule simulated_variants_consequence:
    input:
        "results/simulated/chrom/{chrom}.annot.parquet",
    output:
        "results/simulated/consequence/{consequence}/chrom/{chrom}.parquet",
    wildcard_constraints:
        chrom="|".join(CHROMS),
    run:
        (
            pl.read_parquet(input[0])
            .filter(consequence=wildcards.consequence)
            .write_parquet(output[0])
        )


rule simulated_variants_add_cre:
    input:
        "results/simulated/chrom/{chrom}.annot.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/simulated/chrom_annot_cre/{chrom}.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0])
        V = V.with_columns(original_consequence=pl.col("consequence"))
        cre = pl.read_parquet(input[1])
        V = add_cre(V, cre)
        V = V.drop("original_consequence")
        V.write_parquet(output[0])


rule simulated_variants_add_cre_all:
    input:
        expand("results/simulated/chrom_annot_cre/{chrom}.parquet", chrom=CHROMS),


rule simulated_variants_add_cre_fast:
    input:
        "results/simulated/chrom/{chrom}.annot.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/simulated/chrom_annot_cre_fast/{chrom}.parquet",
    threads: workflow.cores
    run:
        V = pl.read_parquet(input[0])
        V = V.with_columns(original_consequence=pl.col("consequence"))
        cre = pl.read_parquet(input[1])
        V = add_cre_fast(V, cre)
        V = V.drop("original_consequence")
        V.write_parquet(output[0])


rule simulated_variants_add_cre_fast_all:
    input:
        expand("results/simulated/chrom_annot_cre_fast/{chrom}.parquet", chrom=CHROMS),


rule compare_add_cre_implementations:
    """Compare old and new add_cre implementations on a random sample."""
    input:
        "results/simulated/chrom/Y.annot.parquet",
        "results/intervals/cre.parquet",
    output:
        "results/simulated/cre_comparison.txt",
    run:
        import random

        random.seed(42)

        V_full = pl.read_parquet(input[0])
        print(V_full)
        cre = pl.read_parquet(input[1])
        print(cre)

        # Sample 100k variants, then sort for deterministic comparison
        n_sample = 100_000
        indices = sorted(random.sample(range(len(V_full)), min(n_sample, len(V_full))))
        V = V_full[indices]
        print(V)

        # Run both implementations
        V_old = V.with_columns(original_consequence=pl.col("consequence"))
        V_old = add_cre(V_old, cre)
        V_old = V_old.drop("original_consequence")

        V_new = V.with_columns(original_consequence=pl.col("consequence"))
        V_new = add_cre_fast(V_new, cre)
        V_new = V_new.drop("original_consequence")

        # Compare consequence columns by joining on coordinates
        comparison = (
            V_old.select(COORDINATES + ["consequence"])
            .rename({"consequence": "old"})
            .join(
                V_new.select(COORDINATES + ["consequence"]).rename(
                    {"consequence": "new"}
                ),
                on=COORDINATES,
                how="inner",
            )
        )
        match = (comparison["old"] == comparison["new"]).all()
        n_diff = (comparison["old"] != comparison["new"]).sum()

        with open(output[0], "w") as f:
            f.write(f"Sample size: {len(V)}\n")
            f.write(f"Match: {match}\n")
            f.write(f"Differences: {n_diff}\n")
            if not match:
                diff_df = comparison.filter(pl.col("old") != pl.col("new"))
                f.write("\nFirst 20 differences:\n")
                f.write(diff_df.head(20).to_pandas().to_string())

        print(f"Match: {match}, Differences: {n_diff}")
