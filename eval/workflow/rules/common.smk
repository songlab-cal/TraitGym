import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from traitgym.variants import CHROMS, COORDINATES


PRECOMPUTED_MODELS = list(config["precomputed"].keys())
