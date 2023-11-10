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

    n_samples = 100  # number of repeated samples

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
            values = values/np.mean(values)  # normalized to mean

            n_values = np.array(range(1, n_lobuli + 1))


            means = np.zeros_like(n_values, dtype=float)
            medians = np.zeros_like(n_values, dtype=float)
            stds = np.zeros_like(n_values, dtype=float)
            q25s = np.zeros_like(n_values, dtype=float)
            q75s = np.zeros_like(n_values, dtype=float)
            qs = [q25s, q75s]

            for k, n in enumerate(n_values):
                # sample from values and calculate mean
                sample: np.ndarray = random.choice(values, size=n, replace=False, p=None)
                means[k] = sample.mean()
                medians[k] = np.quantile(sample, q=0.5)
                stds[k] = sample.std()
                q25s[k] = np.quantile(sample, q=0.25)
                q75s[k] = np.quantile(sample, q=0.75)

            unit = df[f"{variable}_unit"].unique()[0]
            ax = axes[kvar]
            ax.plot(
                n_values, means,
                # yerr=qs,
                marker=markers[k_subject],
                linestyle="-",
                color="black",
                markerfacecolor=species_colors[species],
                markeredgecolor="black",
                alpha=0.5,
                label=subject,
            )
            # plot the 1.1 and 0.9 line
            ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.0)
            ax.axhline(y=1.1, color='black', linestyle='--', linewidth=1.0)
            ax.set_ylabel(f"Normalized {variable.title()} (-)")
            ax.set_xlabel(f"Number lobulus (-)")

    for ax in axes:
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
    for species in ["mouse", "rat", "pig", "human"]:
        plot_sampling(
            df_all=df,
            species=species,
        )


