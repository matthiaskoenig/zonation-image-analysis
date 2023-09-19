"""
Takes list of polygons and calculates statistics on them.



- area
- number of corners (after useful simplification)
- circumference length
- symmetry

"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
from shapely import Polygon


@dataclass
class LobuleStatistics:

    polygon: Polygon
    area: float


    @classmethod
    def from_polgygon(cls, polygon: Polygon) -> LobuleStatistics:

        area = 1.0

        lobule_statistics = LobuleStatistics(
            polygon=polygon,
            area=area
        )


        return lobule_statistics


    @classmethod
    def to_dataframe(cls, statistics: List[LobuleStatistics]) -> pd.DataFrame:

        df = None

        return df


def visualize_statistics(df_statistics: pd.DataFrame) -> None:
    """Creates histograms or other figures."""
    pass


if __name__ == "__main__":

    # TODO: make this work for a single polygon
    # TODO: make this work for all polygons from a single slide
    # TODO: make this work multiple slides from different classes (species)

    polygons = read_polygons_from_slide()
    statistics_list = [LobuleStatistics.from_polgygon(p) for p in polygons]
    df = LobuleStatistics.to_dataframe(statistics_list)
    visualize_statistics(df_statistics=df)


pass



