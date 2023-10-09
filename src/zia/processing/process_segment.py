import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import shapely
from matplotlib import pyplot as plt
from shapely import LineString, polygonize, GeometryCollection, Polygon, make_valid, polygonize_full

from zia.annotations.annotation.util import PyramidalLevel
from zia.processing.get_segments import LineSegmentsFinder
from zia.processing.lobulus_statistics import LobuleStatistics, SlideStats


def process_line_segments(line_segments: List[List[Tuple[int, int]]],
                          vessel_classes: List[int],
                          vessel_contours: list,
                          final_level: PyramidalLevel) -> SlideStats:
    linestrings = [LineString(s) for s in line_segments]

    vessel_polys = [Polygon(cont.squeeze(axis=1).tolist()) for cont in vessel_contours if len(cont) >=4]
    vessel_polys = [Polygon([(y, x) for x, y in poly.exterior.coords]) for poly in vessel_polys]

    vessel_polys = [p if p.is_valid else make_valid(p) for p in vessel_polys]

    result = shapely.multipolygons(shapely.get_parts(polygonize(linestrings)))

    vessels = []
    lobuli = []

    for poly in result.geoms:
        for vessel_poly in vessel_polys:
            if vessel_poly.buffer(2.0).contains(poly):
                vessels.append(poly)
                break
        else:
            lobuli.append(poly)

    class_0: List[Polygon] = [p for p, c in zip(vessel_polys, vessel_classes) if c == 0]
    class_1: List[Polygon] = [p for p, c in zip(vessel_polys, vessel_classes) if c == 1]

    stats = []
    for lobulus_poly in lobuli:
        c0, c1 = [], []
        c0_idx, c1_idx = [], []
        for i, p in enumerate(class_0):
            if lobulus_poly.contains(p) or lobulus_poly.intersects(p):
                c0.append(p)
                c0_idx.append(i)

        for i, p in enumerate(class_1):
            if lobulus_poly.contains(p) or lobulus_poly.intersects(p):
                c1.append(p)
                c1_idx.append(i)

        stats.append(LobuleStatistics.from_polygon(lobulus_poly, c0, c1, c0_idx, c1_idx))

    meta_data = dict(level=final_level, pixel_size=0.22724690376093626)
    return SlideStats(stats, class_0, class_1, meta_data)


if __name__ == "__main__":
    results_path = Path(__file__).parent
    with open("segmenter.pickle", "rb") as f:
        segmenter: LineSegmentsFinder = pickle.load(f)

    with open("vessels.pickle", "rb") as f:
        classes, contours = pickle.load(f)

    slide_stats = process_line_segments(segmenter.segments_finished,
                                        classes,
                                        contours,
                                        5) # Not important for testing here

    #slide_stats.to_geojson(results_path, "NOR_021")

    slide_stats.plot()
