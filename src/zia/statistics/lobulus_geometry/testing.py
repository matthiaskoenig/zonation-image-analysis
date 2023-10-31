from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd

from zia.statistics.utils.data_provider import SlideStatsProvider


def test_one_way_anova(nomial_var: str, attributes: List[str], data: pd.DataFrame, logs: List[bool]) -> pd.DataFrame:
    results = []
    for attr, log in zip(attributes, logs):
        groups = []
        for group, group_df in data.groupby(nomial_var):
            groups.append(group_df[attr].values)

        if log:
            groups = [np.log10(g) for g in groups]

        stat, pvalue = f_oneway(*groups)

        results.append(dict(test="one-way-anova", nominal_var=nomial_var, attr=attr, log=log, stat=stat, pvalue=pvalue))

    return pd.DataFrame(data=results, index=None)


def test_turkey_hsd(nomial_var: str, attributes: List[str], data: pd.DataFrame, logs: List[bool]) -> pd.DataFrame:
    results = []
    for attr, log in zip(attributes, logs):
        groups = []
        group_names = []
        for group, group_df in data.groupby(nomial_var):
            groups.append(group_df[attr].values)
            group_names.append(group)

        if log:
            groups = [np.log10(g) for g in groups]

        test_result = tukey_hsd(*groups)

        for i in range(0, len(groups)):
            for k in range(i + 1, len(groups)):
                stat = test_result.statistic[i, k]
                pvalue = test_result.pvalue[i, k]
                results.append(dict(test="tukey_hsd",
                                    nominal_var=nomial_var,
                                    attr=attr,
                                    group1=group_names[i],
                                    group2=group_names[k],
                                    log=log,
                                    stat=stat,
                                    pvalue=pvalue))

    return pd.DataFrame(data=results, index=None)


if __name__ == "__main__":
    report_path = SlideStatsProvider.create_report_path("statistical_test_results")

    df = SlideStatsProvider.get_slide_stats_df()

    attributes = ["area", "perimeter", "compactness", "minimum_bounding_radius"]
    logs = [True, True, False, False]

    # species comparison
    anova_results = test_one_way_anova("species", attributes, df, logs)
    anova_results.to_csv(report_path / "anova_species.csv", index=False)
    tukey_kramer_result = test_turkey_hsd("species", attributes, df, logs)
    tukey_kramer_result.to_csv(report_path / "tukey_species.csv", index=False)

    # subject comparison
    for species, species_df in df.groupby("species"):
        annova_results_subjects = test_one_way_anova("subject", attributes, species_df, logs)
        annova_results_subjects.to_csv(report_path / f"anova_{species}_subjects.csv", index=False)

        tukey_kramer_subject = test_turkey_hsd("subject", attributes, species_df, logs)
        tukey_kramer_subject.to_csv(report_path / f"tukey_{species}_subjects.csv", index=False)

        # mouse roi comparison ->TODO: need to know which roi needs to be compared to which roi.
        if species == "mouse":
            pass
