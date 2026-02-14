"""Plotting utilities for TraitGym."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_model_bars(
    ax: Axes,
    subset_data: pd.DataFrame,
    models_config: dict[str, Any],
    show_labels: bool = True,
    baseline: float | None = None,
) -> None:
    """
    Plot horizontal bar chart with error bars for model comparison.

    Args:
        ax: Matplotlib axes to plot on
        subset_data: DataFrame with columns 'model', 'score', 'se', 'n_pos', 'n_neg'
                     Should be pre-sorted by score (ascending for top=best)
        models_config: Config dict mapping model names to {alias, color}
        show_labels: Whether to show y-axis labels (model names)
        baseline: x-axis left limit (random baseline). If None, uses n_pos/(n_pos+n_neg).
    """
    y_positions = range(len(subset_data))
    colors = [
        models_config.get(m, {}).get("color", f"C{i % 10}")
        for i, m in enumerate(subset_data["model"])
    ]
    labels = [
        models_config.get(m, {}).get("alias", m) for m in subset_data["model"]
    ]

    # Create horizontal bar plot
    ax.barh(y_positions, subset_data["score"], color=colors)

    # Add error bars
    ax.errorbar(
        x=subset_data["score"],
        y=y_positions,
        xerr=subset_data["se"],
        fmt="none",
        color="black",
        capsize=0,
    )

    ax.set_yticks(list(y_positions))
    if show_labels:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([])

    if baseline is not None:
        ax.set_xlim(left=baseline)
    else:
        n_pos = subset_data["n_pos"].iloc[0]
        n_neg = subset_data["n_neg"].iloc[0]
        ax.set_xlim(left=n_pos / (n_pos + n_neg))

    sns.despine(ax=ax)


def get_subset_order_and_aliases(
    subsets_config: list[dict[str, str]],
) -> tuple[list[str], dict[str, str]]:
    """
    Extract subset order and aliases from config.

    Args:
        subsets_config: List of dicts with 'name' and 'alias' keys

    Returns:
        Tuple of (ordered list of subset names, dict mapping name to alias)
    """
    subset_order = [s["name"] for s in subsets_config]
    subset_aliases = {s["name"]: s["alias"] for s in subsets_config}
    return subset_order, subset_aliases
