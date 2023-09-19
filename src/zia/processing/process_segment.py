import pickle

import numpy as np
from matplotlib import pyplot as plt
from shapely import LineString, polygonize, GeometryCollection, Polygon, make_valid

from zia.processing.get_segments import LineSegmentsFinder

if __name__ == "__main__":
    with open("segmenter.pickle", "rb") as f:
        segmenter: LineSegmentsFinder = pickle.load(f)

    linestrings = [LineString(s) for s in segmenter.segments_finished]

    with open("vessels.pickle", "rb") as f:
        classes, contours = pickle.load(f)

    vessel_polys = [Polygon(cont.squeeze(axis=1).tolist()) for cont in contours]
    vessel_polys = [Polygon([(y, x) for x, y in poly.exterior.coords]) for poly in vessel_polys]

    print(len(vessel_polys))
    # polygons from linestring: "shapely" Polygons
    result: GeometryCollection = polygonize(linestrings)
    print(type(result))
    print(set([type(g) for g in result.geoms]))
    print(len(result.geoms))
    vessels = []
    lobuli = []

    for poly in result.geoms:
        covered = False
        for vessel_poly in vessel_polys:
            if not vessel_poly.is_valid:
                print("not valid")
                vessel_poly = make_valid(vessel_poly)
            if vessel_poly.buffer(2.0).contains(poly):
                vessels.append(poly)
                covered = True
        if not covered:
            lobuli.append(poly)

    print(len(vessels))
    print(len(lobuli))

    class_0 = [p for p, c in zip(vessel_polys, classes) if c == 0]
    class_1 = [p for p, c in zip(vessel_polys, classes) if c == 1]

    fig, ax = plt.subplots(1, 1, dpi=600)
    colors = np.random.rand(len(lobuli), 3)  # Random RGB values between 0 and 1
    for i, poly in enumerate(lobuli):
        x, y = poly.exterior.xy
        ax.fill(y, x, facecolor=colors[i])

    for i, poly in enumerate(class_0):
        x, y = poly.buffer(1.0).exterior.xy
        ax.fill(y, x, facecolor="black", edgecolor="black", linewidth=0.2)

    for i, poly in enumerate(class_1):
        x, y = poly.buffer(1.0).exterior.xy
        ax.fill(y, x, facecolor="white", edgecolor="black", linewidth=0.2)
    # ax.set_xlim(right=labels.shape[1])
    # ax.set_ylim(top=labels.shape[0])

    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.show()
    exit(0)

    fig, ax = plt.subplots(dpi=600)
    colors = np.random.rand(len(segs), 3)  # Random RGB values between 0 and 1
    for i, line in enumerate(segs):
        x, y = zip(*line)
        ax.plot(y, x, marker="none", color=colors[i], linewidth=0.2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.show()
