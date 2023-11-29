"""Variability plot for the analysis."""
import numpy as np
import pandas as pd
from zia.console import console
from matplotlib import pyplot as plt
from numpy import random


def plot_df(df: pd.DataFrame) -> None:
    """Test plot to access the data."""
    f, axes = plt.subplots(nrows=1, ncols=4, dpi=300, figsize=(20, 5))

    for kplot, species in enumerate(["mouse", "rat", "pig", "human"]):
        df_species = df[df.species == species]
        ax = axes[kplot]
        subjects = df_species.subject.unique()
        console.print(subjects)
        for subject in subjects:
            df_subject = df_species[df_species.subject == subject]
            ax.plot(df_subject.area, df_subject.perimeter, label=subject,
                    linestyle="", marker="o",
                    markeredgecolor="black")

        ax.legend()

    plt.show()


species_colors = {
    'mouse': '#77aadd',
    'rat': '#ee8866',
    'pig': '#dddddd',
    'human': '#44bb99'
}


def plot_sampling(df_all: pd.DataFrame, species: str):
    """Analyse the effect of sampling"""

    n_samples = 1000  # number of repeated samples

    variables = ["area", "perimeter", "compactness", "minimum_bounding_radius"]
    f, axes = plt.subplots(nrows=4, ncols=1, dpi=300, figsize=(5, 20))

    markers = [
        "o", "s", "^", "v", "<", ">"
    ]

    df_species = df_all[(df_all.species == species)]
    subjects = sorted(df_species.subject.unique())
    for k_subject, subject in enumerate(subjects):
        console.rule(title=f"{species} - {subject}")
        df = df_species[df_species.subject == subject]
        n_lobuli = len(df)

        for kvar, variable in enumerate(variables):
            console.print(variable)

            # calculate the values based on sampled lobuli
            values = df[variable].values
            values = values / np.mean(values)  # normalized to mean

            n_values = np.array(range(1, n_lobuli + 1))

            means = np.zeros_like(n_values, dtype=float)
            medians = np.zeros_like(n_values, dtype=float)
            sds = np.zeros_like(n_values, dtype=float)
            ses = np.zeros_like(n_values, dtype=float)
            q05s = np.zeros_like(n_values, dtype=float)
            q25s = np.zeros_like(n_values, dtype=float)
            q75s = np.zeros_like(n_values, dtype=float)
            q95s = np.zeros_like(n_values, dtype=float)
            qs = [q05s, q95s]  # [q25s, q75s]

            for k_lobuli, n_lobuli in enumerate(n_values):
                all_means = np.zeros(shape=(n_samples,))
                # sample from values and calculate mean
                # FIXME: reuse sample for more robust determination
                sample: np.ndarray = random.choice(values, size=n_lobuli, replace=False,
                                                   p=None)
                for k_sample in range(n_samples):
                    all_means[k_sample] = np.mean(sample)

                means[k_lobuli] = np.mean(all_means)
                # medians[k] = np.quantile(sample, q=0.5)
                sds[k_lobuli] = np.std(all_means)
                ses[k_lobuli] = np.std(all_means) / np.sqrt(n_samples)
                q05s[k_lobuli] = np.quantile(all_means, q=0.05)
                q25s[k_lobuli] = np.quantile(all_means, q=0.25)
                q75s[k_lobuli] = np.quantile(all_means, q=0.75)
                q95s[k_lobuli] = np.quantile(all_means, q=0.95)

            ax = axes[kvar]
            if kvar == 0:
                ax.set_title(species.title(), fontweight="bold")

            ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.0)
            ax.axhline(y=1.1, color='black', linestyle='--', linewidth=1.0)
            ax.plot(
                n_values, means,
                # yerr=[means-q05s, q95s-means],
                marker=markers[k_subject],
                linestyle="-",
                color="black",
                markerfacecolor=species_colors[species],
                markeredgecolor="black",
                alpha=0.8,
                label=subject,
            )
            ax.plot(
                n_values, q05s,
                marker="",
                linestyle="-",
                color="black",
                label="Q05" if kvar == 0 else "__nolabel__"
            )
            ax.plot(
                n_values, q25s,
                marker="",
                linestyle="-",
                color="darkgrey",
                label="Q25" if kvar == 0 else "__nolabel__"
            )
            ax.plot(
                n_values, q75s,
                marker="",
                linestyle="-",
                color="darkgrey",
                label="Q75" if kvar == 0 else "__nolabel__"
            )
            ax.plot(
                n_values, q95s,
                marker="",
                linestyle="-",
                color="darkgrey",
                label="Q95" if kvar == 0 else "__nolabel__"
            )

            # plot the 1.1 and 0.9 line
            ax.set_ylabel(f"Normalized {variable.title()} (-)")
            ax.set_xlabel(f"Number lobulus (-)")

    for ax in axes:
        ax.set_ylim(0, 2)
        ax.set_xlim(0.5, 1000)
        # ax.set_xscale("log")
        ax.legend()

    f.savefig(f"{species}_variability.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # load dataframe with the information
    df = pd.read_csv("slide_statistics_df.csv")
    print(df.columns)
    print(df["compactness"])
    console.print(df)

    # plot_df(df)
    # for species in ["mouse", "rat", "pig", "human"]:
    for species in ["mouse"]:
        plot_sampling(
            df_all=df,
            species=species,
        )
