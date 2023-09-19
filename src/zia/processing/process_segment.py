import pickle

import numpy as np
from matplotlib import pyplot as plt

from zia.processing.get_segments import LineSegmentsFinder

if __name__ == "__main__":
    with open("segmenter.pickle", "rb") as f:
        segmenter: LineSegmentsFinder = pickle.load(f)

    m = np.zeros(shape=(len(segmenter.segments_finished), len(segmenter.segments_finished)))

    for i, s1 in enumerate(segmenter.segments_finished):
        for k, s2 in enumerate(segmenter.segments_finished):
            if s1[-1] == s2[0]:
                m[i, k] = 1

    segs = [s for i, s in enumerate(segmenter.segments_finished) if np.any(m[i])]

    fig, ax = plt.subplots(dpi=600)
    colors = np.random.rand(len(segs), 3)  # Random RGB values between 0 and 1
    for i, line in enumerate(segs):
        x, y = zip(*line)
        ax.plot(y, x, marker="none", color=colors[i], linewidth=0.2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.show()


