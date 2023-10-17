import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from zia.statistics.utils.data_provider import SlideStatsProvider


def fractal(p, d, k):
    return 2 / d * (p - k)


if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()

    gb = df.groupby("species")

    fig, ax = plt.subplots(dpi=300)
    ax: plt.Axes

    markers = ["o", "v", "^", "s", "p", "P", "*", "h"]

    for species, color in zip(SlideStatsProvider.species_order, SlideStatsProvider.colors):
        species_df = gb.get_group(species)

        marker_iter = iter(markers)

        for subject, subject_df in species_df.groupby("subject"):
            ax.scatter(subject_df["perimeter"],
                       subject_df["area"],
                       color=color,
                       alpha=0.4,
                       marker=next(marker_iter)
                       )
    x, y = np.log10(df["perimeter"]), np.log10(df["area"])
    pop, pcov = curve_fit(fractal, x, y)

    x_plot = np.linspace(np.min(x), np.max(x), 50)

    y_plot = fractal(x_plot, pop[0], pop[1])

    ax.plot(10 ** x_plot, 10 ** y_plot, color="black", marker="none")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylabel("area (µm$^2$)")
    ax.set_xlabel("perimeter (µm)")
    print(pop)

    plt.show()
