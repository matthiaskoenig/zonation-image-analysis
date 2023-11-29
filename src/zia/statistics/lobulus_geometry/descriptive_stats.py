from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd, levene, kruskal
from scikit_posthocs import posthoc_dunn

from zia.statistics.utils.data_provider import SlideStatsProvider


def descriptive_stats(nomial_var: str, attributes: List[str], data_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for attr in attributes:
        for group, group_df in data_df.groupby(nomial_var):
            data = group_df[attr].values

            unit = set(group_df[f"{attr}_unit"]).pop()

            results.append(
                dict(nominal_var=nomial_var,
                     group=group,
                     attr=attr,
                     unit=unit,
                     mean=np.mean(data),
                     std=np.std(data),
                     se=np.mean(data) / np.sqrt(len(data)),
                     median=np.median(data),
                     min=np.min(data),
                     max=np.max(data),
                     q1=np.percentile(data, q=25),
                     q3=np.percentile(data, q=75),
                     n=len(group_df)
                     )
            )

    df = pd.DataFrame(data=results, index=None)

    return df


def generate_descriptive_stats(slide_stats_df: pd.DataFrame,
                               report_path: Path,
                               attributes: List[str],
                               mouse_lobe_dict: Dict[str, str]):
    with pd.ExcelWriter(report_path / "descriptive-stats.xlsx") as writer:
        # species comparison
        stats = descriptive_stats("species", attributes, slide_stats_df)

        stats.sort_values(by="group", key=lambda column: column.map(lambda e: SlideStatsProvider.species_order.index(e)), inplace=True)
        stats.to_excel(writer, "species-comparison", index=False)

        # subject comparison
        for species, species_df in slide_stats_df.groupby("species"):
            stats_subjects = descriptive_stats("subject", attributes, species_df)
            stats_subjects.to_excel(writer, f"subject-comparison-{species}", index=False)

            # mouse roi comparison
            if species == "mouse":
                for subject, subject_df in species_df.groupby("subject"):
                    roi_dict = {k.split("_")[1]: v for k, v in mouse_lobe_dict.items() if str(subject) in k}

                    stats_mouse_rois = descriptive_stats("roi", attributes, subject_df)
                    stats_mouse_rois["group"] = stats_mouse_rois["group"].map(roi_dict)
                    stats_mouse_rois.to_excel(writer, f"mouse-lobe-comparison-{subject}", index=False)
