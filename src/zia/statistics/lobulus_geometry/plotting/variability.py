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
            values = values/np.mean(values)  # normalized to mean

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
                label="Q05" if kvar==0 else "__nolabel__"
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

def plot_n_geometric(df: pd.DataFrame, distances) -> None:
    """Plot the n required for geometric calculation."""
    f: plt.Figure
    f, axes = plt.subplots(nrows=1, ncols=4, dpi=300, figsize=(17, 4))
    f.subplots_adjust(wspace=0.3)

    for kplot, attr in enumerate(["perimeter", "area", "compactness", "minimum_bounding_radius"]):
        ax = axes[kplot]
        ax.set_title(attr, fontdict={"fontweight": "bold"})
        ax.set_ylabel("Number lobuli", fontdict={"fontweight": "bold"})
        ax.set_xlabel("Margin of error [%]", fontdict={"fontweight": "bold"})
        # ax.set_ylim(top=np.max(df[f"n{distances[0]}"].values))
        ax.set_ylim(top=60)
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


if __name__ == "__main__":
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
    for d in distances:
        key = f"n{d}"

        # 95%
        df[key] = 1 / ((df["mean"]*d)** 2 / (1.96 ** 2 * df["std"] ** 2) + 1 / df["n"])

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


    # # bootstrapping approach
    # # load dataframe with the information
    # df = pd.read_csv("slide_statistics_df.csv")
    # print(df.columns)
    # print(df["compactness"])
    # console.print(df)
    #
    # # plot_df(df)
    # # for species in ["mouse", "rat", "pig", "human"]:
    # for species in ["mouse"]:
    #     plot_sampling(
    #         df_all=df,
    #         species=species,
    #     )


