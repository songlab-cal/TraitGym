"""Tests for traitgym.metrics module."""

import polars as pl
from sklearn.metrics import average_precision_score

from traitgym.metrics import stratified_bootstrap_se


class TestStratifiedBootstrapSE:
    def test_returns_positive_se(self) -> None:
        """SE should be positive for non-trivial data."""
        labels = pl.Series([True] * 50 + [False] * 50)
        scores = pl.Series([0.8] * 30 + [0.4] * 20 + [0.3] * 30 + [0.6] * 20)
        se = stratified_bootstrap_se(average_precision_score, labels, scores)
        assert se > 0

    def test_perfect_predictions_low_se(self) -> None:
        """Perfect separation should have low SE."""
        labels = pl.Series([True] * 50 + [False] * 50)
        scores = pl.Series([1.0] * 50 + [0.0] * 50)
        se = stratified_bootstrap_se(average_precision_score, labels, scores)
        assert se < 0.05  # Should be very low

    def test_reproducible_with_same_n_bootstraps(self) -> None:
        """Same inputs should give same results (seeded)."""
        labels = pl.Series([True] * 50 + [False] * 50)
        scores = pl.Series([0.7] * 25 + [0.3] * 25 + [0.4] * 25 + [0.6] * 25)
        se1 = stratified_bootstrap_se(average_precision_score, labels, scores, n_bootstraps=50)
        se2 = stratified_bootstrap_se(average_precision_score, labels, scores, n_bootstraps=50)
        assert se1 == se2

    def test_more_bootstraps_changes_precision(self) -> None:
        """Different n_bootstraps may give slightly different results."""
        labels = pl.Series([True] * 100 + [False] * 100)
        scores = pl.Series(
            [0.6 + i * 0.001 for i in range(100)] + [0.4 - i * 0.001 for i in range(100)]
        )
        se_50 = stratified_bootstrap_se(average_precision_score, labels, scores, n_bootstraps=50)
        se_100 = stratified_bootstrap_se(
            average_precision_score, labels, scores, n_bootstraps=100
        )
        # Both should be reasonable positive values
        assert se_50 > 0
        assert se_100 > 0
