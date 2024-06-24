from typing import Dict

import numpy as np
import pandas as pd


def create_descriptive_stats(data: pd.Series) -> Dict[str, float]:
    min_ = data.min()
    max_ = data.max()
    mean = data.mean()
    median = data.median()
    std = data.std()
    geo_mean = np.exp(np.mean(np.log(data)))
    q1, q3 = np.percentile(data, [25, 75])

    return dict(
        mean=mean,
        median=median,
        min=min_,
        max=max_,
        std=std,
        geo_mean=geo_mean,
        q1=q1,
        q3=q3,
    )


def create_stats_from_data_dict(data_dict: Dict[str, Dict[str, Dict[str, pd.Series]]]) -> pd.DataFrame:
    stats = []

    for species, species_dict in data_dict.items():
        for group, group_dict in species_dict.items():
            for attr, attr_data in group_dict.items():
                row = dict(species=species, group=group, attr=attr)
                row.update(create_descriptive_stats(attr_data))
                stats.append(row)

    return pd.DataFrame(stats)
