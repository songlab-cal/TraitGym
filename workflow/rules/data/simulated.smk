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
    df = pl.DataFrame({'ref': list(seq)})

    # Add a 'pos' column with the 1-based position using the modern API.
    df = df.with_row_index('pos', offset=1)

    # Define the mapping from each reference base to its possible alternatives.
    valid_bases = {'A', 'C', 'G', 'T'}
    alt_map = {
        'A': ['C', 'G', 'T'],
        'C': ['A', 'G', 'T'],
        'G': ['A', 'C', 'T'],
        'T': ['A', 'C', 'G']
    }

    # Filter the DataFrame to include only rows with valid DNA bases.
    df = df.filter(pl.col('ref').is_in(list(valid_bases)))

    # Use the recommended 'replace_strict' for fast, native mapping.
    # This is the efficient and correct replacement for this pattern.
    df = df.with_columns(
        pl.col('ref').replace_strict(alt_map).alias('alt')
    )

    # "Explode" the 'alt' column to give each SNV its own row.
    df = df.explode('alt')

    # Add the constant 'chrom' column.
    df = df.with_columns(pl.lit(chrom).alias('chrom'))

    # Select and reorder columns to the final format.
    V = df.select(['chrom', 'pos', 'ref', 'alt'])

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