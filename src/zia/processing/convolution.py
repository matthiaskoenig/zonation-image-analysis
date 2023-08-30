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

from zia.processing.filtering import invert_image, filter_img

numcodecs.register_codec(Jpeg2k)
import cv2

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.FOUR

if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        if i in ["HE", "CYP2D6"]:
            continue
        arrays[i] = np.array(a.get(f"{level}"))

    conv = {i: invert_image(a) for i, a in arrays.items()}

    merged = np.stack(list(conv.values()), axis=-1)

    # remove non overlapping pixels
    mask = np.any(merged[:, :, :] == 0, axis=-1)
    merged[mask, :] = 0

    # apply filters

    merged = np.stack([filter_img(merged[:, :, i]) for i in range(merged.shape[2])], axis=-1)

    superpixelslic = cv2.ximgproc.createSuperpixelSLIC(merged, algorithm=cv2.ximgproc.MSLIC, region_size=6)

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

    super_pixel_means = {label: np.mean(merged[labels == label], axis=0) for label in range(num_labels)}

    print(super_pixel_means)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(list(super_pixel_means.values()))

    cluster_centers = kmeans.cluster_centers_
    cluster_distances = [np.sqrt(np.dot(np.array(center).T, np.array(center))) for center in cluster_centers]
    sorted_label_idx = np.argsort(cluster_distances)

    print(np.argmax(cluster_distances))

    print(cluster_distances)

    sp_kmeans_map = {sp: km for sp, km in zip(super_pixel_means.keys(), kmeans.labels_)}
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    lookup = np.vectorize(sp_kmeans_map.get)

    kmeans_repr = lookup(labels)

    plot_pic(kmeans_repr, cmap="Reds")

    template = np.zeros_like(merged[:, :, 0])

    template[kmeans_repr == sorted_label_idx[1]] = 255

    plot_pic(template)

    kernel = np.ones((3, 3), np.uint8)

    eroded = template

    for i in range(10):
        eroded = cv2.erode(eroded, kernel)
        contours, hierarchy = cv2.findContours(eroded.reshape(eroded.shape[0], eroded.shape[1], 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ct_template = np.zeros_like(eroded, dtype=np.uint8)
        cv2.drawContours(ct_template, contours, -1, 255, 2)
        #plot_pic(eroded)
        plot_pic(ct_template)

    # print(superpixel_medians)

    # plot_pic(sp_mask)
    """ fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=600)

    for key, ax in zip(arrays, axes.flatten()):
        print(conv[key].shape)
        final = np.ma.masked_where(mask, conv[key])
        ax.imshow(final.data, cmap="binary_r")
        ax: plt.Axes
        ax.set_title(key)

    plt.show()"""
