from pathlib import Path

import polars as pl
import pytest

from traitgym.precomputed import scan_precomputed_parquet
from traitgym.variants import COORDINATES


class TestScanPrecomputedParquet:
    def test_single_chrom(self, tmp_path: Path) -> None:
        chrom_dir = tmp_path / "chrom"
        chrom_dir.mkdir()
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T", 0.5],
                ["1", 200, "C", "G", 0.8],
            ],
            schema=COORDINATES + ["score"],
            orient="row",
        )
        df.write_parquet(chrom_dir / "1.parquet")

        result = scan_precomputed_parquet(tmp_path, chroms=["1"]).collect()
        assert result.shape[0] == 2
        assert result["score"].to_list() == [0.5, 0.8]

    def test_multiple_chroms(self, tmp_path: Path) -> None:
        chrom_dir = tmp_path / "chrom"
        chrom_dir.mkdir()
        df1 = pl.DataFrame(
            [["1", 100, "A", "T", 0.5]],
            schema=COORDINATES + ["score"],
            orient="row",
        )
        df2 = pl.DataFrame(
            [["2", 200, "C", "G", 0.8]],
            schema=COORDINATES + ["score"],
            orient="row",
        )
        df1.write_parquet(chrom_dir / "1.parquet")
        df2.write_parquet(chrom_dir / "2.parquet")

        result = scan_precomputed_parquet(tmp_path, chroms=["1", "2"]).collect()
        assert result.shape[0] == 2
        assert set(result["chrom"].to_list()) == {"1", "2"}

    def test_raises_on_missing_dir(self) -> None:
        with pytest.raises(FileNotFoundError, match="Precomputed directory not found"):
            scan_precomputed_parquet("/nonexistent/path")

    def test_raises_on_no_parquet_files(self, tmp_path: Path) -> None:
        chrom_dir = tmp_path / "chrom"
        chrom_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            scan_precomputed_parquet(tmp_path, chroms=["1"])
