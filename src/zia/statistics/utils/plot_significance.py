from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_significance(ax: plt.Axes, groups: List[str], p_table: pd.DataFrame, log: bool, critical_p=0.01):
    # Get info about y-axis
    bottom, top = ax.get_ylim()
    if log:
        bottom, top = np.log10(bottom), np.log10(top)

    yrange = top - bottom

    pairs = []
    p_values = []
    for i in range(len(groups)):
        for k in range(i + 1, len(groups)):

            g1, g2 = groups[i], groups[k]

            print(g1, g2)

            p = p_table[((p_table.group1 == g1) & (p_table.group2 == g2)) | ((p_table.group1 == g2) & (p_table.group2 == g1))].iloc[0]["pvalue"]

            # Columns corresponding to the datasets of interest
            x1 = i + 1
            x2 = k + 1
            # What level is this bar among the bars above the plot?
            level = i
            # Plot the bar

            if p < critical_p:
                pairs.append([x1, x2])
                p_values.append(p)

    levels = [[]]
    size = len(groups)

    sorted_zipped = sorted(zip(pairs, p_values), key=lambda xy: xy[0][1] - xy[0][0])

    sorted_pairs = [pair for pair, _ in sorted_zipped]
    sorted_pvalues = [pvalue for _, pvalue in sorted_zipped]

    for pair in sorted_pairs:
        level = levels[-1]
        if len(level) == 0:
            level.append(pair)
        elif pair[0] >= level[-1][1] and pair[1] <= size:
            level.append(pair)
        else:
            new_level = [pair]
            levels.append(new_level)

    for i, level in enumerate(levels):
        for pair in level:
            p = sorted_pvalues.pop(0)
            x1, x2 = pair
            bar_height = (yrange * 0.08 * i) + top

            bar_tips = bar_height - (yrange * 0.02)
            if log:
                bar_tips = 10 ** bar_tips
                bar_height = 10 ** bar_height
            ax.plot(
                [(x1 + 0.025), (x1 + 0.025), (x2 - 0.025), (x2 - 0.025)],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            if p < 1e-4:
                sig_symbol = '***'
            elif p < 1e-3:
                sig_symbol = '**'
            elif p < critical_p:
                sig_symbol = '*'
            else:
                sig_symbol = "NS"

            text_height = bar_height

            if log:
                text_height = 10 ** (np.log10(bar_height))

            ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k', fontsize=7)

        # Adjust y-axis
    bottom, top = ax.get_ylim()

    if log:
        log_bottom, log_top = np.log(bottom), np.log(top)
        log_range = log_top - log_bottom
        new_log_min = log_bottom - 0.02 * log_range

        ax.set_ylim(bottom=np.exp(new_log_min), top=top)

    else:
        ax.set_ylim(bottom - 0.02 * yrange, top)
