import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def ez_corr_heatmap(df: pd.DataFrame, vmin: float = None,
                    vmax: float = None, center: float = 0):
    """
    Custom Heatmap Plot
    References: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    Inputs:pandas Dataframe
    Optional Inputs: vmin, vmax, center
    Output: Seaborn Heatmap Plot
    """
    corr = df.corr(numeric_only=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 11, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                center=center, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})