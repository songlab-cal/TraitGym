"""Tests for traitgym.metrics module."""

import polars as pl
from sklearn.metrics import average_precision_score

from traitgym.metrics import (
    iid_mean_se,
    mean_reciprocal_rank_scores,
    pairwise_accuracy_scores,
    stratified_bootstrap_se,
)


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


class TestPairwiseAccuracyScores:
    def test_perfect_predictions(self) -> None:
        labels = pl.Series([True, False, True, False])
        scores = pl.Series([1.0, 0.0, 1.0, 0.0])
        match_groups = pl.Series([0, 0, 1, 1])
        result = pairwise_accuracy_scores(labels, scores, match_groups)
        assert result.to_list() == [1.0, 1.0]

    def test_wrong_predictions(self) -> None:
        labels = pl.Series([True, False, True, False])
        scores = pl.Series([0.0, 1.0, 0.0, 1.0])
        match_groups = pl.Series([0, 0, 1, 1])
        result = pairwise_accuracy_scores(labels, scores, match_groups)
        assert result.to_list() == [0.0, 0.0]

    def test_ties(self) -> None:
        labels = pl.Series([True, False])
        scores = pl.Series([0.5, 0.5])
        match_groups = pl.Series([0, 0])
        result = pairwise_accuracy_scores(labels, scores, match_groups)
        assert result.to_list() == [0.5]

    def test_mean_is_accuracy(self) -> None:
        labels = pl.Series([True, False, True, False, True, False])
        scores = pl.Series([0.9, 0.1, 0.3, 0.7, 0.5, 0.5])
        match_groups = pl.Series([0, 0, 1, 1, 2, 2])
        result = pairwise_accuracy_scores(labels, scores, match_groups)
        # Group 0: 1.0, Group 1: 0.0, Group 2: 0.5
        assert result.mean() == 0.5


class TestMeanReciprocalRankScores:
    def test_positive_ranked_first(self) -> None:
        labels = pl.Series([True, False, False])
        scores = pl.Series([1.0, 0.5, 0.2])
        match_groups = pl.Series([0, 0, 0])
        result = mean_reciprocal_rank_scores(labels, scores, match_groups)
        assert result.to_list() == [1.0]

    def test_positive_ranked_last(self) -> None:
        labels = pl.Series([True, False, False])
        scores = pl.Series([0.1, 0.5, 0.9])
        match_groups = pl.Series([0, 0, 0])
        result = mean_reciprocal_rank_scores(labels, scores, match_groups)
        assert result.to_list() == [1.0 / 3.0]

    def test_multiple_groups(self) -> None:
        labels = pl.Series([True, False, True, False])
        scores = pl.Series([1.0, 0.0, 0.0, 1.0])
        match_groups = pl.Series([0, 0, 1, 1])
        result = mean_reciprocal_rank_scores(labels, scores, match_groups)
        # Group 0: rank 1 -> 1.0, Group 1: rank 2 -> 0.5
        assert result.to_list() == [1.0, 0.5]
        assert result.mean() == 0.75

    def test_1_to_9_matching(self) -> None:
        labels = pl.Series([True] + [False] * 9)
        scores = pl.Series([0.5] + [0.1 * i for i in range(9, 0, -1)])
        match_groups = pl.Series([0] * 10)
        result = mean_reciprocal_rank_scores(labels, scores, match_groups)
        assert len(result) == 1


class TestIidMeanSe:
    def test_constant_values(self) -> None:
        values = pl.Series([0.5, 0.5, 0.5])
        mean, se = iid_mean_se(values)
        assert mean == 0.5
        assert se == 0.0

    def test_known_values(self) -> None:
        values = pl.Series([1.0, 0.0])
        mean, se = iid_mean_se(values)
        assert mean == 0.5
        import math
        assert abs(se - values.std() / math.sqrt(2)) < 1e-10
