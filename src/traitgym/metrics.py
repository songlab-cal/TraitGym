"""Metrics utilities for TraitGym."""

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
