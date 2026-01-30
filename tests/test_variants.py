import polars as pl
import pytest

from traitgym.variants import (
    CHROMS,
    COORDINATES,
    NUCLEOTIDES,
    filter_chroms,
    filter_snp,
)


class TestConstants:
    def test_coordinates(self) -> None:
        assert COORDINATES == ["chrom", "pos", "ref", "alt"]

    def test_nucleotides(self) -> None:
        assert NUCLEOTIDES == ["A", "C", "G", "T"]

    def test_chroms(self) -> None:
        expected = [str(i) for i in range(1, 23)] + ["X", "Y"]
        assert CHROMS == expected
        assert len(CHROMS) == 24


class TestFilterSnp:
    def test_keeps_valid_snps(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "C", "G"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 2

    def test_filters_invalid_ref(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "N", "G"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1
        assert result["ref"][0] == "A"

    def test_filters_invalid_alt(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "C", "-"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1
        assert result["alt"][0] == "T"

    def test_filters_indels(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "AT", "A"],
                ["1", 300, "G", "GC"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1


class TestFilterChroms:
    def test_keeps_valid_chroms(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["22", 200, "C", "G"],
                ["X", 300, "G", "A"],
                ["Y", 400, "T", "C"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_chroms(df)
        assert result.shape[0] == 4

    def test_filters_invalid_chroms(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["MT", 200, "C", "G"],
                ["chr1", 300, "G", "A"],
                ["23", 400, "T", "C"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_chroms(df)
        assert result.shape[0] == 1
        assert result["chrom"][0] == "1"

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame(schema={"chrom": pl.String, "pos": pl.Int64, "ref": pl.String, "alt": pl.String})
        result = filter_chroms(df)
        assert result.shape[0] == 0
