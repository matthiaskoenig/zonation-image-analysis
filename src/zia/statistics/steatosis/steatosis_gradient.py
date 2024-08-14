from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from zia.statistics.steatosis.utils.get_data import DIET
from zia.statistics.utils.data_provider import capitalize


def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W HFD"
    return "Steatosis"


def map_diet_on_df(df: pd.DataFrame) -> None:
    df["group"] = list(map(map_to_group, df['species'], df['diet']))


def plot_species_comparison_gradient(report_path: Path,
                                     steatosis_portality_df: pd.DataFrame,
                                     control_portality_df: pd.DataFrame
                                     ):
    plt.style.use("tableau-colorblind10")
    group_order = {
        "human": ["Control", "Steatosis"],
        "mouse": ["Control", "2W HFD", "4W HFD"],
        "rat": ["Control", "2W HFD", "4W HFD"],
    }
    protein_order = ["HE", "GS", "CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4"]

    colors = ["#77AADD", "#EE8866", "#44BB99"]
    markers = ["o", "P", "^"]
    control_portality_df["group"] = "Control"

    steatosis_portality_df["diet"] = steatosis_portality_df["subject"].map(DIET)
    map_diet_on_df(steatosis_portality_df)

    distance_df_all = pd.concat([control_portality_df, steatosis_portality_df])

    for species, color in zip(["mouse", "rat", "human"], colors):
        distance_df = distance_df_all[distance_df_all["species"] == species]

        group_gb = distance_df.groupby("group")

        fig, axes = plt.subplots(nrows=len(protein_order) + 1, ncols=len(group_gb) + 1, dpi=300,
                                 figsize=(4 * 2, len(protein_order) * 1.85),
                                 layout="constrained")

        medians_array = np.empty(shape=(len(protein_order), len(group_gb)), dtype=object)

        for col, group in enumerate(group_order[species]):

            species_df = group_gb.get_group(group)
            protein_gb = species_df.groupby("protein")

            for row, protein in enumerate(protein_order):
                ax: plt.Axes = axes[row, col]
                protein_df = protein_gb.get_group(protein.lower())

                bins = np.histogram_bin_edges(protein_df["pv_dist"], range=(0, 1), bins=12)
                binned, bins = pd.cut(protein_df["pv_dist"], bins=bins, retbins=True)

                x = []
                y = []
                medians = []

                for i in range(len(bins) - 1):
                    df_bin = protein_df[(protein_df["pv_dist"] > bins[i]) & (protein_df["pv_dist"] <= bins[i + 1])]
                    x.append((bins[i] + bins[i + 1]) / 2)
                    y.append(df_bin["intensity"])
                    medians.append(np.median(df_bin["intensity"]))

                medians_array[row, col] = (x, medians)
                d = (bins[1] - bins[0])

                bp = ax.boxplot(x=y, positions=x, widths=d, patch_artist=True, showfliers=False, whis=(5, 95))

                ax.plot(x, medians,
                        marker=markers[col],
                        markerfacecolor=color,
                        markeredgecolor="black",
                        linewidth=1,
                        markersize=4,
                        zorder=10,
                        color="black")

                for box in bp["boxes"]:
                    box.set(facecolor=color, linewidth=0.5)
                for box in bp["medians"]:
                    box.set(color="None", linewidth=0)
                for box in bp["caps"]:
                    box.set(linewidth=0.5)
                for box in bp["whiskers"]:
                    box.set(linewidth=0.5)

                lobule_count = len(protein_df.groupby(["subject", "roi", "lobule"]))

                ax.text(x=0.02, y=0.98, s=f"n={lobule_count}", fontsize=10, ha="left", va="top", transform=ax.transAxes)

                ax.set_xticks([])

        # plot all species per protein
        for i in range(len(protein_order)):
            for k, (x, medians) in enumerate(medians_array[i, :]):
                axes[i, -1].plot(x, medians, marker=markers[k], color=color, markeredgecolor="black",
                                 markersize=4)

        # plot all proteins per group
        for i in range(len(group_gb)):
            for k, (x, medians) in enumerate(medians_array[:, i]):
                axes[-1, i].plot(x, medians, marker="o", color=f"C{k}", markeredgecolor="black",
                                 markersize=4)

        fig: plt.Figure
        fig.supxlabel("Portality (-)", fontsize=14)
        fig.supylabel("Normalized intensity (-)", fontsize=14)

        for ax in axes[-1, :].flatten():
            ax.xaxis.set_ticks([0, 1], labels=["PP", "PV"])

        axes[-1, -1].set_axis_off()

        for i, group in enumerate(group_gb):
            for ax in axes[:, i].flatten():
                ax.set_xlim(left=0, right=1)

        for ax in axes.flatten():
            ax.set_ylim(top=1.2, bottom=-0.05)

        for ax in axes[:-1, 1:].flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        for ax in axes[:-1, 0].flatten():
            ax.set_xticklabels([])

        for ax in axes[-1, 1:].flatten():
            ax.set_yticklabels([])

        for protein, ax in zip(protein_order, axes[:-1, -1].flatten()):
            ax.set_ylabel(protein, fontsize=14, fontweight="bold")
            ax.yaxis.set_label_position("right")

        for group, ax in zip(group_order[species], axes[0, :].flatten()):
            ax.set_title(capitalize(group), fontsize=14, fontweight="bold")

        axes[-2, -1].xaxis.set_ticks([0, 1], labels=["PP", "PV"])

        handles = []

        for i, protein in enumerate(protein_order):
            handles.append(Line2D([], [], linestyle="-", color=f"C{i}", marker="o", markeredgecolor="black", label=protein))

        axes[-1, -1].legend(handles=handles, frameon=False, ncols=1, prop=dict(size=10))

        fig.savefig(report_path / f"expression-gradient-steatosis-{species}.png", dpi=600)
        fig.savefig(report_path / f"expression-gradient-steatosis-{species}.pdf", dpi=600)
        fig.savefig(report_path / f"expression-gradient-steatosis-{species}.svg", dpi=600)

        plt.show()
