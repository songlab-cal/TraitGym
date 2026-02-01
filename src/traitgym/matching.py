"""Feature-based sample matching utilities."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from traitgym.variants import COORDINATES

MATCH_GROUP_COL = "match_group"


def match_features(
    pos: pl.DataFrame,
    neg: pl.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    k: int,
    scale: bool = True,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Match positive samples to k negative samples based on features.

    For each unique combination of categorical features, finds the k closest
    negative samples for each positive sample based on continuous features.

    Args:
        pos: Positive samples DataFrame with COORDINATES columns.
        neg: Negative samples DataFrame with COORDINATES columns.
        continuous_features: Columns to use for distance-based matching.
        categorical_features: Columns for exact matching (grouping).
        k: Number of negative samples to match per positive sample.
        scale: Whether to scale continuous features before matching.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with matched samples, match_group column, and sorted by COORDINATES.

    Raises:
        ValueError: If required columns are missing from pos or neg.
    """
    required_cols = COORDINATES + continuous_features + categorical_features
    _validate_columns(pos, required_cols, "pos")
    _validate_columns(neg, required_cols, "neg")
    all_features = continuous_features + categorical_features
    _validate_no_nulls(pos, all_features, "pos")
    _validate_no_nulls(neg, all_features, "neg")

    pos_pd = pos.to_pandas()
    neg_pd = neg.to_pandas()

    if scale and len(continuous_features) > 0:
        pos_pd, neg_pd, match_cols = _scale_features(
            pos_pd, neg_pd, continuous_features
        )
    else:
        match_cols = continuous_features

    pos_pd = pos_pd.set_index(categorical_features)
    neg_pd = neg_pd.set_index(categorical_features)

    pos_list: list[pd.DataFrame] = []
    neg_list: list[pd.DataFrame] = []

    for group_key in pos_pd.index.drop_duplicates():
        result = _match_single_group(pos_pd, neg_pd, group_key, match_cols, k, seed)
        if result is not None:
            pos_group, neg_group = result
            pos_list.append(pos_group)
            neg_list.append(neg_group)

    result_df = _combine_results(pos_list, neg_list, k)

    if scale and len(continuous_features) > 0:
        scaled_cols = [f"{c}_scaled" for c in continuous_features]
        result_df = result_df.drop(columns=scaled_cols, errors="ignore")

    return _sort_by_coordinates(pl.from_pandas(result_df))


def _validate_columns(
    df: pl.DataFrame,
    required: list[str],
    name: str,
) -> None:
    """Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate.
        required: List of required column names.
        name: Name of DataFrame for error messages.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_no_nulls(
    df: pl.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    """Validate DataFrame has no null values in specified columns.

    Args:
        df: DataFrame to validate.
        columns: List of column names to check.
        name: Name of DataFrame for error messages.

    Raises:
        ValueError: If any columns contain null values.
    """
    for col in columns:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"{name} has {null_count} null values in column '{col}'")


def _scale_features(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply RobustScaler to features, return scaled DataFrames and column names.

    Args:
        pos: Positive samples DataFrame.
        neg: Negative samples DataFrame.
        features: Column names to scale.

    Returns:
        Tuple of (scaled pos, scaled neg, list of scaled column names).
    """
    scaler = RobustScaler()
    all_data = pd.concat([pos[features], neg[features]])
    scaler.fit(all_data)

    scaled_cols = [f"{c}_scaled" for c in features]
    pos = pos.copy()
    neg = neg.copy()
    pos[scaled_cols] = scaler.transform(pos[features])
    neg[scaled_cols] = scaler.transform(neg[features])

    return pos, neg, scaled_cols


def _match_single_group(
    pos_indexed: pd.DataFrame,
    neg_indexed: pd.DataFrame,
    group_key: tuple | str,
    match_cols: list[str],
    k: int,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Match positives to negatives within a single categorical group.

    Args:
        pos_indexed: Positive samples indexed by categorical features.
        neg_indexed: Negative samples indexed by categorical features.
        group_key: The categorical group key to match.
        match_cols: Columns to use for distance matching.
        k: Number of negatives per positive.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (matched positives, matched negatives) or None if no negatives.
    """
    pos_group = pos_indexed.loc[[group_key]].reset_index()

    try:
        neg_group = neg_indexed.loc[[group_key]].reset_index()
    except KeyError:
        warnings.warn(f"No negatives found for category: {group_key}")
        return None

    n_pos_needed = len(pos_group)
    n_neg_available = len(neg_group)
    n_neg_needed = n_pos_needed * k

    if n_neg_needed > n_neg_available:
        warnings.warn(
            f"Insufficient negatives for category {group_key}: "
            f"need {n_neg_needed}, have {n_neg_available}. Subsampling positives."
        )
        n_pos_possible = n_neg_available // k
        pos_group = pos_group.sample(n_pos_possible, random_state=seed)

    if len(match_cols) == 0:
        n_neg_to_sample = len(pos_group) * k
        neg_matched = neg_group.sample(n_neg_to_sample, random_state=seed)
    else:
        neg_matched = _find_closest(pos_group, neg_group, match_cols, k)

    return pos_group, neg_matched


def _find_closest(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    cols: list[str],
    k: int,
) -> pd.DataFrame:
    """Find k closest negatives for each positive using Euclidean distance.

    Sampling is done without replacement: once a negative is selected,
    it cannot be selected again.

    Args:
        pos: Positive samples DataFrame.
        neg: Negative samples DataFrame.
        cols: Columns to use for distance calculation.
        k: Number of closest negatives to find per positive.

    Returns:
        DataFrame with k negatives for each positive.
    """
    dist_matrix = cdist(pos[cols], neg[cols])
    closest_indices: list[int] = []

    for i in range(len(pos)):
        sorted_indices = np.argsort(dist_matrix[i])[:k].tolist()
        closest_indices.extend(sorted_indices)
        dist_matrix[:, sorted_indices] = np.inf

    return neg.iloc[closest_indices]


def _combine_results(
    pos_list: list[pd.DataFrame],
    neg_list: list[pd.DataFrame],
    k: int,
) -> pd.DataFrame:
    """Combine matched groups and assign match_group column.

    Args:
        pos_list: List of matched positive DataFrames.
        neg_list: List of matched negative DataFrames.
        k: Number of negatives per positive.

    Returns:
        Combined DataFrame with match_group column.
    """
    if not pos_list:
        return pd.DataFrame()

    all_pos = pd.concat(pos_list, ignore_index=True)
    all_pos[MATCH_GROUP_COL] = np.arange(len(all_pos))

    all_neg = pd.concat(neg_list, ignore_index=True)
    all_neg[MATCH_GROUP_COL] = np.repeat(all_pos[MATCH_GROUP_COL].values, k)

    return pd.concat([all_pos, all_neg], ignore_index=True)


def _sort_by_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    """Sort DataFrame by genomic coordinates.

    Args:
        df: DataFrame with COORDINATES columns.

    Returns:
        DataFrame sorted by chrom, pos, ref, alt (chrom as string).
    """
    return df.sort(COORDINATES)
