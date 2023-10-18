import matplotlib.pyplot as plt
import numpy as np


def plot_significance(ax: plt.Axes, p_table: np.ndarray, log: bool):
    # Get info about y-axis
    bottom, top = ax.get_ylim()
    if log:
        bottom, top = np.log10(bottom), np.log10(top)

    yrange = top - bottom

    for i in range(p_table.shape[0]):
        for k in range(i + 1, p_table.shape[1]):
            # Columns corresponding to the datasets of interest
            x1 = i + 1
            x2 = k + 1
            # What level is this bar among the bars above the plot?

            if k - i == 1:
                level = 1
            elif k - i == 2:
                level = 2 + i
            else:
                level = 4
            # Plot the bar
            bar_height = (yrange * 0.08 * level) + top

            bar_tips = bar_height - (yrange * 0.02)
            if log:
                bar_tips = 10 ** bar_tips
                bar_height = 10 ** bar_height
            plt.plot(
                [(x1 + 0.025), (x1 + 0.025), (x2 - 0.025), (x2 - 0.025)],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
            # Significance level
            p = p_table[i, k]
            print(p)
            if p < 0.001:
                sig_symbol = '***'
            elif p < 0.01:
                sig_symbol = '**'
            elif p < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = "NS"
            text_height = bar_height + (yrange * 0.01)
            if log:
                text_height = 10 ** (np.log10(bar_height) + (yrange * 0.01))

            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')

        # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    """# Annotate sample size below each box
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom, fr'n = {sample_size}', ha='center', size='x-small')"""
