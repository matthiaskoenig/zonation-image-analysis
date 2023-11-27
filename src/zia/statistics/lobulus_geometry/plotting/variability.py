"""Variability plot for the analysis.

Bootstrapping analysis

TODO: reuse samples; I.e. same subset of samples for different n;
bootstrapping samples with replacement.
It is a straightforward way to derive estimates of standard errors and confidence intervals for complex estimators of the distribution
Bootstrap 95% confidence intervals?

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

sample size calculator for the mean:
https://www.statology.org/sample-size-calculator-for-a-mean/

The sample size required to estimate a population mean with a certain level of confidence and a desired margin of error is calculated as:
Sample size =(zα/2σ/E)2

https://www.omnicalculator.com/statistics/sample-size
    How accurate should your result be? (margin of error)
    What level of confidence do you need? (confidence level)
    What is your initial estimate? (proportion estimate)

https://online.stat.psu.edu/stat506/lesson/2/2.1

n = 1 / (0.1827**2/(1.96**2 * 0.253**2) + 1/1580) ~ 7.3
n = 1 / (0.1911**2/(1.96**2 * 0.324**2) + 1/664) ~ 10.8

n = 1 / (0.2027**2/(1.96**2 * 0.3779**2) + 1/176) ~ 12.4

"""
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from zia.console import console
from matplotlib import pyplot as plt
from numpy import random

species_colors = {
    'mouse': '#77aadd',
    'rat': '#ee8866',
    'pig': '#dddddd',
    'human': '#44bb99'
}


def plot_n_geometric(df: pd.DataFrame, distances) -> None:
    """Plot the n required for geometric calculation."""
    f: plt.Figure
    f, axes = plt.subplots(nrows=1, ncols=4, dpi=300, figsize=(17, 4))
    f.subplots_adjust(wspace=0.3)

    for kplot, attr in enumerate(["perimeter", "area", "compactness", "minimum_bounding_radius"]):



        ax = axes[kplot]
        ax.set_title(attr, fontdict={"fontweight": "bold", "fontsize": 16})
        ax.set_ylabel("Number of lobuli [-]", fontdict={"fontweight": "bold", "fontsize": 14})
        ax.set_xlabel("Margin of error [%]", fontdict={"fontweight": "bold", "fontsize": 14})
        # ax.set_ylim(top=np.max(df[f"n{distances[0]}"].values))
        ax.set_ylim(top=60)
        # ax.set_xlim(left=0)
        ax.fill([8, 12, 12, 8], [0, 0, 60, 60], 'gray', alpha=0.2, edgecolor=None,
                label="__nolabel__")

        ax.set_xticks(distances * 100)

        for species in ["mouse", "rat", "pig", "human"]:
            df_species = df[(df.species == species) & (df.attr == attr)]
            # collect data
            means = []
            stds = []
            for d in distances:

                values = df_species[f"n{d}"].values
                means.append(np.mean(values))
                stds.append(np.std(values))

            ax.errorbar(
                x=distances*100,  # conversion to percent
                y=means,
                yerr=stds,
                label=species,
                color=species_colors[species],
                linestyle="-", marker="o",
                markeredgecolor="black"
            )

        ax.legend()

    plt.show()
    f.savefig("n_lobuli.png", bbox_inches="tight")


def analysis_n_geometric():
    """Analysis for the required n for the geometric parameters."""
    # statistics approach
    # 1. read dataframe
    data = pd.read_excel("descriptive-stats.xlsx", sheet_name=None)
    dfs = []
    for key in [
        "subject-comparison-human",
        "subject-comparison-pig",
        "subject-comparison-rat",
        "subject-comparison-mouse",
    ]:
        df = data[key]
        species = key.split("-")[-1]
        df["nominal_var"] = species
        df.rename(columns={"nominal_var": "species", "group": "subject"}, inplace=True)
        df.drop(["log", "se", "median", "min", "max", "q1", "q3"], axis=1, inplace=True)
        # df.drop(["log"], axis=1, inplace=True)
        dfs.append(df)

    df = pd.concat(dfs)

    # 2. calculate statistics (n_lobule) for 99%, 95% and 90%
    distances = np.linspace(start=0.05, stop=0.35, num=7)
    # distances = np.insert(distances, 0, 0.01)
    for d in distances:
        key = f"n{d}"

        # 95%
        df[key] = 1 / (
                (df["mean"] * d) ** 2 / (1.96 ** 2 * df["std"] ** 2) + 1 / df["n"])

    console.print(df)

    # sort
    df.species = pd.Categorical(
        df.species,
        categories=["human", "pig", "rat", "mouse"],
        ordered=True
    )
    df.attr = pd.Categorical(
        df.attr,
        categories=["perimeter", "area", "compactness", "minimum_bounding_radius"],
        ordered=True
    )
    df.sort_values(by=["attr", "species"], inplace=True)

    # store as xlsx
    df.to_excel("n_lobuli.xlsx", sheet_name="data", index=False)

    # 3. create plot of the statistics
    plot_n_geometric(df, distances)


def plot_n_gradient(df: pd.DataFrame):
    plt.style.use("tableau-colorblind10")

    protein_order = ["HE", "GS", "CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4"]
    colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]


    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(5, 10), layout="constrained")


    # Simple example:
    # mouse - GS - 12 bins pv_dist
    color = species_colors["mouse"]

    species_protein_df = df[(df.species == "mouse") & (df.protein == "GS")]
    # analysis by individual subjects

    n_bins = 12

    x_all: Dict[str, np.ndarray] = {}
    medians_all: Dict[str, np.ndarray] = {}
    means_all: Dict[str, np.ndarray] = {}
    stds_all: Dict[str, np.ndarray] = {}
    ns_all: Dict[str, np.ndarray] = {}

    for (subject, protein_df) in species_protein_df.groupby(["subject"]):

        lobule_count = len(protein_df.groupby(["subject", "roi", "lobule"]))
        console.print(f"{lobule_count=}")

        # binning
        bins = np.histogram_bin_edges(protein_df["pv_dist"], range=(0, 1), bins=n_bins)
        binned, bins = pd.cut(protein_df["pv_dist"], bins=bins, retbins=True)

        x = np.zeros(shape=(n_bins-1, ))
        medians = np.zeros(shape=(n_bins-1, ))
        means = np.zeros(shape=(n_bins-1, ))
        stds = np.zeros(shape=(n_bins-1, ))
        ns = np.zeros(shape=(n_bins-1, ))

        for i in range(n_bins - 1):
            df_bin = protein_df[(protein_df["pv_dist"] > bins[i]) & (protein_df["pv_dist"] <= bins[i + 1])]
            x[i] = (bins[i] + bins[i + 1]) / 2
            medians[i] = np.median(df_bin["nintensity"])
            means[i] = np.mean(df_bin["nintensity"])
            stds[i] = np.std(df_bin["nintensity"])

            # TODO: calculate this for the different slides
            # calculation of the required n at 95% confidence and ME=0.1
            d = 0.1
            ns[i] = 1 / ((means[i] * d) ** 2 / (1.96 ** 2 * stds[i] ** 2) + 1 / lobule_count)

        x_all[subject] = x
        medians_all[subject] = medians
        means_all[subject] = means
        stds_all[subject] = stds
        ns_all[subject] = ns

        # plot subject curves
        ax1.plot(
            x, medians,
            marker="o",
            markerfacecolor=color,
            markeredgecolor="black",
            linewidth=1,
            markersize=4,
            zorder=10,
            color="black",
            alpha=0.5
        )

        ax2.plot(x, ns,
             marker="o",
             markerfacecolor=color,
             markeredgecolor="black",
             linewidth=1,
             markersize=4,
             zorder=10,
             color="black",
             alpha=0.5,
        )

    # plot mean +- sd


    for ax in [ax1, ax2]:
        # ax.text(x=0.98, y=0.01, s=f"n={lobule_count}", fontsize=10, ha="right", va="bottom", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_xlim(left=0, right=1)

    ax1.set_ylim(bottom=0, top=1.1)
    ax2.set_ylim(bottom=0)

            # ax.set_xticklabels([])
            # ax.set_yticklabels([])



    plt.show()


if __name__ == "__main__":
    # analysis_n_geometric()

    # analysis of the required n for determining the zonation
    # load data for the different regions
    csv_lobule_distances = "/home/mkoenig/Downloads/manuscript/distance-data/lobule_distances.csv"
    pkl_lobule_distances = "/home/mkoenig/Downloads/manuscript/distance-data/lobule_distances.pkl"

    read = False
    if read:
        df = pd.read_csv(csv_lobule_distances)
        # normalization based on max intensity values of one roi
        norm_dfs = []
        for (subject, roi, protein), roi_df in df.groupby(["subject", "roi", "protein"]):
            max_intensity = np.percentile(roi_df["intensity"], 99)
            roi_df["nintensity"] = roi_df["intensity"] / max_intensity
            norm_dfs.append(roi_df)
        df = pd.concat(norm_dfs)
        df.to_pickle(pkl_lobule_distances)

    df = pd.read_pickle(pkl_lobule_distances)
    console.print(df.columns)
    console.print(df)

    # calculate and plot the n
    plot_n_gradient(df=df)
