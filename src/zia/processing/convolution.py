import matplotlib.pyplot as plt
import numpy as np
import zarr

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.data_store import ZarrGroups
from imagecodecs.numcodecs import Jpegxl, Jpeg2k
import numcodecs
from sklearn.cluster import KMeans

numcodecs.register_codec(Jpeg2k)
import cv2

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.FOUR


def convolute_meadian(img: np.ndarray, ksize=3):
    median_blurr = cv2.medianBlur(img, ksize)
    return median_blurr[::2, ::2]


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        if i in ["he", "cyp2d6"]:
            continue
        img = np.array(a.get(f"{level}"))
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        arrays[i] = img  # cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, mask=mask)

    conv = {i: convolute_meadian(convolute_meadian(a)) for i, a in arrays.items()}

    merged = np.stack(list(conv.values()), axis=-1)

    superpixelslic = cv2.ximgproc.createSuperpixelSLIC(merged, region_size=6)

    superpixelslic.getNumberOfSuperpixels()
    # print(superpixelslic.getLabels())

    superpixelslic.iterate(num_iterations=20)

    mask = superpixelslic.getLabelContourMask()

    # print(superpixelslic.getNumberOfSuperpixels())

    # plot_pic(superpixelslic.getLabelContourMask())

    # Get the labels and number of superpixels
    # print(set(superpixelslic.getLabels().flatten()))
    labels = superpixelslic.getLabels()
    num_labels = superpixelslic.getNumberOfSuperpixels()

    # Create a mask for each superpixel
    sp_mask = np.zeros(shape=merged.shape[:2])

    merged = merged.astype(float)

    superpixel_medians = {label: np.median(merged[labels == label], axis=0) for label in range(num_labels)}

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(list(superpixel_medians.values()))

    cluster_centers = kmeans.cluster_centers_
    cluster_distances = [np.sqrt(np.dot(np.array(center).T, np.array(center))) for center in cluster_centers]
    print(cluster_distances)

    sp_kmeans_map = {sp: km for sp, km in zip(superpixel_medians.keys(), kmeans.labels_)}
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    lookup = np.vectorize(sp_kmeans_map.get)

    kmeans_repr = lookup(labels)

    plot_pic(kmeans_repr, cmap="Reds")

    # print(superpixel_medians)

    # plot_pic(sp_mask)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=600)

    for key, ax in zip(arrays, axes.flatten()):
        print(conv[key].shape)
        final = np.ma.masked_where(mask, conv[key])
        ax.imshow(final.data, cmap="binary_r")
        ax: plt.Axes
        ax.set_title(key)

    plt.show()
