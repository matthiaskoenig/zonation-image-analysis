import pickle
from pathlib import Path
from typing import List

import numpy as np
import shapely
from matplotlib import pyplot as plt
from shapely import LineString, polygonize, GeometryCollection, Polygon, make_valid, polygonize_full

from zia.processing.get_segments import LineSegmentsFinder
from zia.processing.lobulus_statistics import LobuleStatistics, SlideStats

if __name__ == "__main__":
    results_path = Path(__file__).parent
    with open("segmenter.pickle", "rb") as f:
        segmenter: LineSegmentsFinder = pickle.load(f)

    linestrings = [LineString(s) for s in segmenter.segments_finished]

    with open("vessels.pickle", "rb") as f:
        classes, contours = pickle.load(f)

    vessel_polys = [Polygon(cont.squeeze(axis=1).tolist()) for cont in contours]
    vessel_polys = [Polygon([(y, x) for x, y in poly.exterior.coords]) for poly in vessel_polys]

    vessel_polys = [p if p.is_valid else make_valid(p) for p in vessel_polys]
    print(len(vessel_polys))
    # polygons from linestring: "shapely" Polygons
    result = shapely.multipolygons(shapely.get_parts(polygonize(linestrings)))
    print(type(result))
    print(set([type(g) for g in result.geoms]))
    print(len(result.geoms))
    vessels = []
    lobuli = []

    for poly in result.geoms:
        covered = False
        for vessel_poly in vessel_polys:
            if vessel_poly.buffer(2.0).contains(poly):
                vessels.append(poly)
                break
        else:
            lobuli.append(poly)

    print(len(vessels))
    print(len(lobuli))

    class_0: List[Polygon] = [p for p, c in zip(vessel_polys, classes) if c == 0]
    class_1: List[Polygon] = [p for p, c in zip(vessel_polys, classes) if c == 1]

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

        stats.append(LobuleStatistics.from_polgygon(lobulus_poly, c0, c1, c0_idx, c1_idx))

    slide_stats = SlideStats(stats, class_0, class_1)
    slide_stats.to_geojson(results_path, "NOR_021")

    slide_stats.plot()

    plt.show()
