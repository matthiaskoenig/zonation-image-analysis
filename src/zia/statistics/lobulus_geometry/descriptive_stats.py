from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd, levene, kruskal
from scikit_posthocs import posthoc_dunn

from zia.statistics.utils.data_provider import SlideStatsProvider


def test_levenne(nomial_var: str, attributes: List[str], data: pd.DataFrame, logs: List[bool]) -> pd.DataFrame:
    results = []
    for attr, log in zip(attributes, logs):
        for group, group_df in data.groupby(nomial_var):
            data = group_df[attr].values
            if log:
                data = np.log10(data)
            unit = set(group_df["unit"]).pop()

            results.append(
                dict(test="levenne",
                     nominal_var=nomial_var,
                     attr=attr,
                     log=log,
                     unit=unit,
                     mean=np.mean(data),
                     std=np.mean(data),
                     se=np.mean(data) / np.sqrt(len(data)),
                     median=np.median(data),
                     min=np.min(data),
                     max=np.max(data),
                     q1=np.percentile(data, q=25),
                     q3=np.percentile(data, q=75),
                     )
            )

    return pd.DataFrame(data=results, index=None)


def test_dunns(nomial_var: str, attributes: List[str], data: pd.DataFrame, logs: List[bool]) -> pd.DataFrame:
    results = []
    for attr, log in zip(attributes, logs):
        groups = []
        group_names = []
        for group, group_df in data.groupby(nomial_var):
            groups.append(group_df[attr].values)
            group_names.append(group)

        if log:
            groups = [np.log10(g) for g in groups]

        test_result = posthoc_dunn(groups, p_adjust="bonferroni")

        for i in range(0, len(groups)):
            for k in range(i + 1, len(groups)):
                # stat = test_result[i, k]
                pvalue = test_result.iloc[i, k]
                results.append(dict(test="dunns-post-hoc",
                                    nominal_var=nomial_var,
                                    attr=attr,
                                    group1=group_names[i],
                                    group2=group_names[k],
                                    log=log,
                                    stat=None,
                                    pvalue=pvalue))

    return pd.DataFrame(data=results, index=None)


if __name__ == "__main__":
    report_path = SlideStatsProvider.create_report_path("statistical_test_results")

    df = SlideStatsProvider.get_slide_stats_df()

    attributes = ["area", "perimeter", "compactness", "minimum_bounding_radius"]
    logs = [True, True, False, True]

    # species comparison
    anova_results = test_one_way_anova("species", attributes, df, logs)
    anova_results.to_csv(report_path / "anova_species.csv", index=False)

    anova_results = test_kruskal("species", attributes, df, logs)
    anova_results.to_csv(report_path / "kruskal_species.csv", index=False)

    levennes_result = test_levenne("species", attributes, df, logs)
    levennes_result.to_csv(report_path / "levenne_species.csv", index=False)
    tukey_kramer_result = test_turkey_hsd("species", attributes, df, logs)
    tukey_kramer_result.to_csv(report_path / "tukey_species.csv", index=False)

    dunns_result = test_dunns("species", attributes, df, logs)
    dunns_result.to_csv(report_path / "dunns_species.csv", index=False)

    # subject comparison
    for species, species_df in df.groupby("species"):
        annova_results_subjects = test_one_way_anova("subject", attributes, species_df, logs)
        annova_results_subjects.to_csv(report_path / f"anova_{species}_subjects.csv", index=False)

        kruskal_results_subjects = test_kruskal("subject", attributes, species_df, logs)
        kruskal_results_subjects.to_csv(report_path / f"kruskal_{species}_subjects.csv", index=False)

        levennes_result = test_levenne("subject", attributes, species_df, logs)
        levennes_result.to_csv(report_path / f"levenne_{species}_subject.csv", index=False)

        tukey_kramer_subject = test_turkey_hsd("subject", attributes, species_df, logs)
        tukey_kramer_subject.to_csv(report_path / f"tukey_{species}_subjects.csv", index=False)

        dunns_subject = test_dunns("subject", attributes, species_df, logs)
        dunns_subject.to_csv(report_path / f"dunns_{species}_subjects.csv", index=False)

        # mouse roi comparison
        if species == "mouse":
            for subject, subject_df in species_df.groupby("subject"):
                annova_mouse_rois = test_one_way_anova("roi", attributes, subject_df, logs)
                annova_mouse_rois.to_csv(report_path / f"anova_mouse_rois.csv", index=False)

                kruskal_mouse_rois = test_kruskal("roi", attributes, subject_df, logs)
                kruskal_mouse_rois.to_csv(report_path / f"kruskal_mouse_rois.csv", index=False)

                levennes_result = test_levenne("roi", attributes, subject_df, logs)
                levennes_result.to_csv(report_path / "levenne_mouse_rois.csv", index=False)

                tukey_kramer_mouse_rois = test_turkey_hsd("roi", attributes, subject_df, logs)
                tukey_kramer_mouse_rois.to_csv(report_path / f"tukey_mouse_rois.csv", index=False)

                dunns_mouse_rois = test_dunns("roi", attributes, subject_df, logs)
                dunns_mouse_rois.to_csv(report_path / f"dunns_mouse_rois.csv", index=False)
