from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from scipy.stats._mannwhitneyu import MannwhitneyuResult
from scipy.stats._stats_py import KruskalResult

from zia.statistics.utils.data_provider import SlideStatsProvider


@dataclass(init=True)
class TestResult:
    attr: str
    test: str
    statistic: float
    p_val: float
    groups: List[str]
    log: bool = False


def kruskal_wallis_test(df: pd.DataFrame, groupby: str, attr: str, log=False) -> TestResult:
    data = []
    group_labels = []
    for species, species_df in df.groupby(groupby):
        data.append(species_df[attr])
        group_labels.append(str(species))

    if log:
        data = [np.log10(d) for d in data]

    kruskal_result: KruskalResult = kruskal(*data)

    return TestResult(attr=attr,
                      test="Kruskal-Wallis",
                      statistic=kruskal_result.statistic,
                      p_val=kruskal_result.pvalue,
                      groups=group_labels,
                      log=log)


def mann_whitney_u_test(df: pd.DataFrame, groupby: str, attr: str, log=False) -> List[TestResult]:
    gb = df.groupby(groupby)

    groups = [str(k) for k in gb.groups.keys()]
    print(groups)

    test_results = []

    for i, g1 in enumerate(groups):
        for k, g2 in enumerate(groups[i + 1:]):
            x = gb.get_group(g1)[attr]
            y = gb.get_group(g2)[attr]
            if log:
                x, y = np.log10(x), np.log10(y)
            mann_whitney_u_result: MannwhitneyuResult = mannwhitneyu(x, y)

            test_results.append(
                TestResult(
                    attr=attr,
                    test="Mann-Whitney-U",
                    statistic=mann_whitney_u_result.statistic,
                    p_val=mann_whitney_u_result.pvalue,
                    groups=[g1, g2],
                    log=log
                )
            )

    return test_results


def kruskal_wallis_test_species(df: pd.DataFrame, attributes, is_log) -> Dict[str, List[TestResult]]:
    result_dict = {"kruskal-wallis-inter-species": []}
    for attr, log in zip(attributes, is_log):
        result_dict["kruskal-wallis-inter-species"].append(kruskal_wallis_test(df, "species", attr, log))

    return result_dict


def kruskal_wallis_test_subjects(df: pd.DataFrame, attributes, is_log) -> Dict[str, List[TestResult]]:
    result_dict = {}
    for species, species_df in df.groupby("species"):
        result_dict[f"kruskal-wallis-{species}-subject"] = []
        for attr, log in zip(attributes, is_log):
            result_dict[f"kruskal-wallis-{species}-subject"].append(kruskal_wallis_test(species_df, "subject", attr, log))

    return result_dict


def mann_whitney_u_inter_species(df: pd.DataFrame, attributes, is_log) -> Dict[str, List[TestResult]]:
    result_dict = {"mann-whitney-u-inter-species": []}
    for attr, log in zip(attributes, is_log):
        result_dict["mann-whitney-u-inter-species"].append(mann_whitney_u_test(df, "species", attr, log))

    return result_dict


def mann_whitney_u_test_subjects(df: pd.DataFrame, attributes, is_log) -> Dict[str, List[TestResult]]:
    result_dict = {}
    for species, species_df in df.groupby("species"):
        result_dict[f"mann-whitney-u-{species}-subject"] = []
        for attr, log in zip(attributes, is_log):
            result_dict[f"mann-whitney-u-{species}-subject"].append(mann_whitney_u_test(df, "subjects", attr, log))

    return result_dict


if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    attributes = ["area", "perimeter", "compactness"]
    is_log = [True, True, False]
    result_dict = {}

    result_dict.update(kruskal_wallis_test_species(df, attributes, is_log))
    result_dict.update(mann_whitney_u_inter_species(df, attributes, is_log))
    result_dict.update(kruskal_wallis_test_subjects(df, attributes, is_log))

    print(result_dict.keys())
