import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.config import read_config
from zia.processing.lobulus_statistics import SlideStats
from zia.processing.visualization_species_comparison import merge_to_one_df

from scipy.optimize import curve_fit


def fractal(p, d, k):
    return 2 / d * (p - k)


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_stain_separated = config.image_data_path / "slide_statistics"
    report_path = config.reports_path / "boxplots"
    report_path.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])

    slide_stats = {}

    species_order = ["mouse", "rat", "pig", "human"]

    a = 0.5
    colors = [(102 / 255, 194 / 255, 165 / 255),
              (252 / 255, 141 / 255, 98 / 255),
              (141 / 255, 160 / 255, 203 / 255),
              (231 / 255, 138 / 255, 195 / 255)]

    for subject_dir in subject_dirs:
        subject = subject_dir.stem
        roi_dict = {}
        # print(subject)

        roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
        for roi_dir in roi_dirs:
            roi = roi_dir.stem
            # print(roi)
            roi_dict[roi] = SlideStats.load_from_file_system(roi_dir)

        slide_stats[subject] = roi_dict

    df = merge_to_one_df(slide_stats)
    print(df.columns)

    gb = df.groupby("species")

    fig, ax = plt.subplots(dpi=300)
    ax: plt.Axes

    markers = ["o", "v", "^", "s", "p", "P", "*", "h"]

    for species, color in zip(species_order, colors):
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
