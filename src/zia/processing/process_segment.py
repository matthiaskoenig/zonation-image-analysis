import pickle

import numpy as np
from matplotlib import pyplot as plt
from shapely import LineString, polygonize, GeometryCollection

from zia.processing.get_segments import LineSegmentsFinder

if __name__ == "__main__":
    with open("segmenter.pickle", "rb") as f:
        segmenter: LineSegmentsFinder = pickle.load(f)

    m = np.zeros(shape=(len(segmenter.segments_finished), len(segmenter.segments_finished)))

    for i, s1 in enumerate(segmenter.segments_finished):
        for k, s2 in enumerate(segmenter.segments_finished):
            if s1[-1] == s2[0]:
                m[i, k] = 1

    linestrings = [LineString(s) for s in segmenter.segments_finished]

    # polygons from linestring: "shapely" Polygons
    result: GeometryCollection = polygonize(linestrings)
    print(type(result))
    print(set([type(g) for g in result.geoms]))
    #print(result)
    #print(linestrings)

    fig, ax = plt.subplots(1, 1, dpi=600)
    colors = np.random.rand(len(result.geoms), 3)  # Random RGB values between 0 and 1
    for i, poly in enumerate(result.geoms):
        x, y = poly.exterior.xy
        ax.fill(y, x, facecolor=colors[i])

    #ax.set_xlim(right=labels.shape[1])
    #ax.set_ylim(top=labels.shape[0])

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
