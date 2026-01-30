"""Module for loading precomputed genomic features using lazy Polars."""

from pathlib import Path

import polars as pl

from traitgym.variants import CHROMS, COORDINATES


def scan_precomputed_parquet(
    precomputed_dir: Path | str,
    chroms: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazily scan per-chromosome parquet files for precomputed features.

    Args:
        precomputed_dir: Directory containing chrom/{1-22,X,Y}.parquet files.
        chroms: List of chromosomes to load. If None, loads all chromosomes.

    Returns:
        LazyFrame with columns: chrom, pos, ref, alt, score

    Raises:
        FileNotFoundError: If precomputed_dir does not exist.
    """
    precomputed_dir = Path(precomputed_dir)
    if not precomputed_dir.exists():
        raise FileNotFoundError(f"Precomputed directory not found: {precomputed_dir}")

    if chroms is None:
        chroms = CHROMS

    chrom_dir = precomputed_dir / "chrom"
    parquet_files = [chrom_dir / f"{chrom}.parquet" for chrom in chroms]
    existing_files = [f for f in parquet_files if f.exists()]

    if not existing_files:
        raise FileNotFoundError(f"No parquet files found in {chrom_dir}")

    return pl.concat([pl.scan_parquet(f) for f in existing_files])
