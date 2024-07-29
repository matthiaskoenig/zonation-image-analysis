from typing import Dict

import numpy as np
import pandas as pd


def create_descriptive_stats(data: pd.Series) -> Dict[str, float]:
    min_ = data.min()
    max_ = data.max()
    mean = data.mean()
    median = data.median()
    std = data.std()
    log_std = np.std(np.log10(data))
    cv = mean / std
    log_cv = np.sqrt(np.exp(np.std(np.log(data)) ** 2) - 1)
    geo_mean = np.exp(np.mean(np.log(data)))
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    n = len(data)

    return dict(
        mean=mean,
        median=median,
        iqr=iqr,
        min=min_,
        max=max_,
        std=std,
        geo_mean=geo_mean,
        q1=q1,
        q3=q3,
        log_std=log_std,
        cv=cv,
        log_cv=log_cv,
        n=n
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
