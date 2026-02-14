"""Metrics utilities for TraitGym."""

import math
from typing import Callable

import polars as pl


def stratified_bootstrap_se(
    metric: Callable,
    labels: pl.Series,
    scores: pl.Series,
    n_bootstraps: int = 100,
) -> float:
    """
    Compute standard error using stratified bootstrap sampling by label.

    Samples positives and negatives separately with replacement, ensuring
    class balance is preserved across bootstrap samples.

    Args:
        metric: Scoring function that takes (y_true, y_score) arrays
        labels: Boolean series of true labels
        scores: Series of prediction scores
        n_bootstraps: Number of bootstrap iterations

    Returns:
        Standard error of the metric across bootstrap samples
    """
    df = pl.DataFrame({"label": labels, "score": scores})
    pos_df = df.filter(pl.col("label"))
    neg_df = df.filter(~pl.col("label"))

    bootstraps = []
    for i in range(n_bootstraps):
        pos_sample = pos_df.sample(fraction=1.0, with_replacement=True, seed=i)
        neg_sample = neg_df.sample(fraction=1.0, with_replacement=True, seed=i)
        boot_df = pl.concat([pos_sample, neg_sample])
        bootstraps.append(metric(boot_df["label"], boot_df["score"]))

    return pl.Series(bootstraps).std()


def iid_mean_se(values: pl.Series) -> tuple[float, float]:
    """Compute mean and standard error from i.i.d. per-group scores."""
    n = len(values)
    return values.mean(), values.std() / math.sqrt(n)


def pairwise_accuracy_scores(
    labels: pl.Series, scores: pl.Series, match_groups: pl.Series
) -> pl.Series:
    """Compute per-match-group pairwise accuracy outcomes.

    For each match group (1 positive, 1 negative), returns 1.0 if the positive
    scores higher, 0.5 on tie, 0.0 otherwise.

    Args:
        labels: Boolean series of true labels.
        scores: Series of prediction scores.
        match_groups: Series of match group identifiers.

    Returns:
        Series of per-group outcomes (1.0, 0.5, or 0.0).
    """
    df = pl.DataFrame(
        {"label": labels, "score": scores, "match_group": match_groups}
    )
    pos = df.filter(pl.col("label")).select("match_group", "score").rename({"score": "pos_score"})
    neg = df.filter(~pl.col("label")).select("match_group", "score").rename({"score": "neg_score"})
    paired = pos.join(neg, on="match_group")
    return paired.select(
        pl.when(pl.col("pos_score") > pl.col("neg_score"))
        .then(1.0)
        .when(pl.col("pos_score") == pl.col("neg_score"))
        .then(0.5)
        .otherwise(0.0)
        .alias("outcome")
    )["outcome"]


def pairwise_accuracy(
    labels: pl.Series, scores: pl.Series, match_groups: pl.Series
) -> tuple[float, float]:
    """Compute pairwise accuracy and its standard error.

    Args:
        labels: Boolean series of true labels.
        scores: Series of prediction scores.
        match_groups: Series of match group identifiers.

    Returns:
        Tuple of (pairwise accuracy, standard error).
    """
    return iid_mean_se(pairwise_accuracy_scores(labels, scores, match_groups))


def mean_reciprocal_rank_scores(
    labels: pl.Series, scores: pl.Series, match_groups: pl.Series
) -> pl.Series:
    """Compute per-match-group reciprocal rank of the positive variant.

    For each match group, ranks variants by score descending and returns
    1/rank of the positive variant.

    Args:
        labels: Boolean series of true labels.
        scores: Series of prediction scores.
        match_groups: Series of match group identifiers.

    Returns:
        Series of per-group 1/rank values.
    """
    df = pl.DataFrame(
        {"label": labels, "score": scores, "match_group": match_groups}
    )
    ranked = df.with_columns(
        pl.col("score")
        .rank(method="average", descending=True)
        .over("match_group")
        .alias("rank")
    )
    pos_ranks = ranked.filter(pl.col("label"))
    return pos_ranks.select((1.0 / pl.col("rank")).alias("rr"))["rr"]


def mean_reciprocal_rank(
    labels: pl.Series, scores: pl.Series, match_groups: pl.Series
) -> tuple[float, float]:
    """Compute mean reciprocal rank and its standard error.

    Args:
        labels: Boolean series of true labels.
        scores: Series of prediction scores.
        match_groups: Series of match group identifiers.

    Returns:
        Tuple of (MRR, standard error).
    """
    return iid_mean_se(mean_reciprocal_rank_scores(labels, scores, match_groups))
