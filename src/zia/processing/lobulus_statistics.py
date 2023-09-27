"""
Takes list of polygons and calculates statistics on them.



- area
- number of corners (after useful simplification)
- circumference length
- symmetry

"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional

import geojson
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry import shape


@dataclass(init=True)
class SlideStats:
    lobule_stats: List[LobuleStatistics]
    vessels_central: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_portal: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    meta_data: dict

    def to_geojson(self, result_dir: Path) -> None:
        result_dir.mkdir(parents=True, exist_ok=True)
        features = [geojson.Feature(geometry=g.__geo_interface__) for g in self.vessels_central]
        col = geojson.FeatureCollection(features=features)

        with open(result_dir / "central_vessels.geojson", "w") as f:
            geojson.dump(col, f)

        features = [geojson.Feature(geometry=g.__geo_interface__) for g in self.vessels_portal]
        col = geojson.FeatureCollection(features=features)
        with open(result_dir / "portal_vessels.geojson", "w") as f:
            geojson.dump(col, f)

        lobule_features = []
        for i, lobule_stat in enumerate(self.lobule_stats):
            lf: geojson.Feature = lobule_stat.to_geo_json_feature_collection()
            lobule_features.append(lf)

        col = geojson.FeatureCollection(features=lobule_features)
        col["metaData"] = self.meta_data
        with open(result_dir / "lobuli.geojson", "w") as f:
            geojson.dump(col, f)

    @classmethod
    def load_from_file_system(cls, result_dir: Path) -> SlideStats:
        if not result_dir.exists():
            raise FileNotFoundError("The specified location was not found.")

        central_vessel_dir = result_dir / "central_vessels.geojson"
        if not central_vessel_dir.exists():
            raise FileNotFoundError(f"{central_vessel_dir} does not exist.")

        with open(central_vessel_dir, "r") as f:
            col = geojson.load(f)

        central_vessels = [shape(feature["geometry"]) for feature in col["features"]]

        portal_vessel_dir = result_dir / "portal_vessels.geojson"
        if not portal_vessel_dir.exists():
            raise FileNotFoundError(f"{portal_vessel_dir} does not exist.")

        with open(portal_vessel_dir, "r") as f:
            col = geojson.load(f)

        portal_vessels = [shape(feature["geometry"]) for feature in col["features"]]

        lobule_dir = result_dir / "lobuli.geojson"
        if not lobule_dir.exists():
            raise FileNotFoundError(f"{lobule_dir} does not exist.")

        with open(lobule_dir, "r") as f:
            col = geojson.load(f)

        lobule_stats = []

        for feature in col["features"]:
            central_vessels_idx = feature["properties"]["central_vessels"]
            portal_vessels_idx = feature["properties"]["portal_vessels"]

            lobule_stat = LobuleStatistics.from_polgygon(
                shape(feature["geometry"]),
                [central_vessels[i] for i in central_vessels_idx],
                [portal_vessels[i] for i in portal_vessels_idx],
                central_vessels_idx,
                portal_vessels_idx
            )

            lobule_stats.append(lobule_stat)

        return cls(lobule_stats, central_vessels, portal_vessels, col.get("meta_data"))

    def plot(self):
        fig, ax = plt.subplots(1, 1, dpi=600)
        colors = np.random.rand(len(self.lobule_stats), 3)  # Random RGB values between 0 and 1
        for i, stat in enumerate(self.lobule_stats):
            x, y = stat.polygon.exterior.xy
            ax.fill(y, x, facecolor=colors[i], edgecolor="black", linewidth=0.2)

        for i, geom in enumerate(self.vessels_central):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor="black", edgecolor="black", linewidth=0.2)

        for i, geom in enumerate(self.vessels_portal):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor="white", edgecolor="black", linewidth=0.2)
        # ax.set_xlim(right=labels.shape[1])
        # ax.set_ylim(top=labels.shape[0])

        ax.set_aspect("equal")
        ax.invert_yaxis()

        plt.show()

    def plot_on_axis(self, ax: plt.Axes,
                     lobulus_ec: str = "lime",
                     lobulus_fc: Optional[str] = None,
                     lobulus_alpha: float = 1,
                     pvessel_ec: str = "magenta",
                     pvessel_fc: Optional[str] = "magenta",
                     pvessel_alpha: float = 0.5,
                     cvessel_ec: str = "cyan",
                     cvessel_fc: Optional[str] = "cyan",
                     cvessel_alpha: float = 0.5
                     ):

        for i, stat in enumerate(self.lobule_stats):
            x, y = stat.polygon.exterior.xy
            ax.fill(y, x, facecolor=lobulus_fc if lobulus_fc is not None else "none", edgecolor=lobulus_ec, alpha=lobulus_alpha, linewidth=1)

        for i, geom in enumerate(self.vessels_central):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor=cvessel_fc, edgecolor=cvessel_ec, alpha=cvessel_alpha, linewidth=1)

        for i, geom in enumerate(self.vessels_portal):
            x, y = geom.buffer(1.0).exterior.xy
            ax.fill(y, x, facecolor=pvessel_fc, edgecolor=pvessel_ec, alpha=pvessel_alpha, linewidth=1)


    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for stat in self.lobule_stats:
            row_dict = dict(
                area=stat.get_area(),
                perimeter=stat.get_perimeter(),
                n_central_vessel=len(stat.vessels_central),
                n_portal_vessel=len(stat.vessels_portal),
                central_vessel_cross_section=stat.get_central_vessel_cross_section(),
                portal_vessel_cross_section=stat.get_portal_vessel_cross_section(),
                compactness=stat.get_compactness(),
                area_without_vessels=stat.get_poly_area_without_vessel_area()
            )
            rows.append(row_dict)
        return pd.DataFrame(rows)


@dataclass
class LobuleStatistics:
    polygon: Polygon
    vessels_central: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_portal: List[Union[Polygon, MultiPolygon, GeometryCollection]]
    vessels_central_idx: List[int]
    vessels_portal_idx: List[int]

    @classmethod
    def from_polgygon(cls, polygon: Polygon,
                      vessels_central: List[Polygon],
                      vessels_portal: List[Polygon],
                      vessels_central_idx: List[int],
                      vessels_portal_idx: List[int]
                      ) -> LobuleStatistics:
        lobule_statistics = LobuleStatistics(
            polygon=polygon,
            vessels_central=vessels_central,
            vessels_portal=vessels_portal,
            vessels_central_idx=vessels_central_idx,
            vessels_portal_idx=vessels_portal_idx,
        )

        return lobule_statistics

    def get_area(self) -> float:
        return self.polygon.area

    def has_central_vessel(self) -> bool:
        return len(self.vessels_central) != 0

    def has_portal_vessel(self) -> bool:
        return len(self.vessels_portal) != 0

    def get_central_vessel_cross_section(self) -> float:
        if not self.has_central_vessel():
            return 0.0
        else:
            return sum([g.area for g in self.vessels_central])

    def get_portal_vessel_cross_section(self) -> float:
        if not self.has_portal_vessel():
            return 0.0
        else:
            return sum([g.area for g in self.vessels_portal])

    def get_poly_area_without_vessel_area(self) -> float:
        intersection_area = 0.0
        for vp in self.vessels_portal + self.vessels_central:
            if self.polygon.contains(vp):
                intersection_area += vp.area
            else:
                intersection = self.polygon.intersection(vp)
                intersection_area += intersection.area

        return self.get_area() - intersection_area

    def get_perimeter(self) -> float:
        return self.polygon.length

    def get_compactness(self) -> float:
        """isoperimetric quotient -> ratio of the area of the shape to the area of a circle with the same perimeter"""
        area = self.polygon.area
        perimeter = self.get_perimeter()

        return area * 4 * np.pi / perimeter ** 2

    def to_geo_json_feature_collection(self) -> geojson.Feature:
        return geojson.Feature(geometry=self.polygon.__geo_interface__,
                               properties={
                                   "central_vessels": self.vessels_central_idx,
                                   "portal_vessels": self.vessels_portal_idx
                               })

    @classmethod
    def to_dataframe(cls, statistics: List[LobuleStatistics]) -> pd.DataFrame:
        pass




def visualize_statistics(df_statistics: pd.DataFrame) -> None:
    """Creates histograms or other figures."""
    pass


if __name__ == "__main__":
    # TODO: make this work for a single polygon
    # TODO: make this work for all polygons from a single slide
    # TODO: make this work multiple slides from different classes (species)
    result_dir = Path(__file__).parent
    slide_stats = SlideStats.load_from_file_system(result_dir, "NOR_021")

pass
