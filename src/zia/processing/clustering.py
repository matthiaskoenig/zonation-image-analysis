import pickle
from typing import List

import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpeg2k
from shapely import Polygon
from sklearn.cluster import KMeans

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.log import get_logger
from zia.processing.filtering import invert_image, filter_img

numcodecs.register_codec(Jpeg2k)
import cv2

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.FOUR

logger = get_logger(__file__)


def check_if_poly_is_in_any_of_the_lists(poly_to_check: Polygon, lists: List[List[Polygon]]) -> int:
    for i, polygons in enumerate(lists):
        for poly in polygons:
            if poly.contains(poly_to_check):
                return i
    return len(lists)


def get_polys(binary: np.ndarray) -> list[Polygon]:
    # get backgorund contours
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.dilate(binary, kernel)

    contours, hierarchy = cv2.findContours(eroded.reshape(template.shape[0], eroded.shape[1], 1).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    ct_template = np.zeros_like(eroded, dtype=np.uint8)
    cv2.drawContours(ct_template, contours, -1, 255, 2)
    # plot_pic(eroded)
    # plot_pic(ct_template)
    return [Polygon(contour.reshape(-1, 2)) for contour in contours if len(contour) > 3]


def get_and_classify_background_polys(binary: np.ndarray, labels: np.ndarray, sorted_label_idx: np.ndarray, n_clusters: int):
    contours, hierarchy = cv2.findContours(binary.reshape(binary.shape[0], binary.shape[1], 1).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # remove the image frame
    contours.pop(0)

    # remove the tissue boundary
    tissue_boundary = contours.pop(0)

    filtered_contours = []
    # Iterate through the contours
    for contour in contours:
        is_contained = False
        for other_contour in contours:
            if contour is not other_contour:
                if cv2.pointPolygonTest(other_contour, tuple(contour[0][0].astype(float)), False) > 0:
                    # 'contour' is contained within 'other_contour'
                    is_contained = True
                    break
        if not is_contained:
            filtered_contours.append(contour)

    count_vectors = []
    classified_contours = []

    kernel = np.ones((3, 3), np.uint8)
    for cont in filtered_contours:
        # min_h, min_w = np.min(cont, axis=0).flatten()
        # max_h, max_w = np.max(cont, axis=0).flatten()

        # create mask from contour
        ct_template = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(ct_template, [cont], -1, 255, thickness=cv2.FILLED)

        dilated_vessel_mask = cv2.dilate(ct_template, kernel).astype(bool)
        vessel_mask = ct_template.astype(bool)

        diff_mask = vessel_mask ^ dilated_vessel_mask

        boundary_labels = labels[diff_mask]
        boundary_labels = boundary_labels[boundary_labels != None]

        if boundary_labels.size == 0:
            "No foreground labels in proximity found. Dropped contour from analysis."
            continue

        count_vector = []
        for i in range(n_clusters):
            count_vector.append(boundary_labels[boundary_labels == sorted_label_idx[i]].size / boundary_labels.size)

        count_vectors.append(count_vector)
        classified_contours.append(cont)

    kmeans = KMeans(n_clusters=2, n_init="auto")
    kmeans.fit(count_vectors)

    # metric for sorting these clusters
    sorted_idx = np.argsort(np.sum(kmeans.cluster_centers_[:, :2], axis=1))
    classes = [sorted_idx[label] for label in kmeans.labels_]

    to_plot = np.zeros_like(binary, dtype=np.uint8)

    for i, cnt in enumerate(classified_contours):
        class_ = classes[i]
        c = 255 if class_ == 0 else 175
        cv2.drawContours(to_plot, [cnt], -1, c, 2)

    plot_pic(to_plot)

    return classes, classified_contours, tissue_boundary


if __name__ == "__main__":
    n_clusters = 5
    config = read_config(BASE_PATH / "configuration.ini")

    logger.info(f"Load images for subject {subject}")
    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        if i in ["HE"]:
            continue
        arrays[i] = np.array(a.get(f"{level}"))


    conv = {i: invert_image(a) for i, a in arrays.items()}

    merged = np.stack(list(conv.values()), axis=-1)
    logger.info(f"Inverterd and merged images into image stack of shape {merged.shape}")
    # remove non overlapping pixels

    mask = np.any(merged[:, :, :] == 0, axis=-1)
    merged[mask, :] = 0
    # apply filters

    logger.info("Apply image filters.")
    merged = np.stack([filter_img(merged[:, :, i]) for i in range(merged.shape[2])], axis=-1)

    logger.info("Run superpixel algorithm.")
    superpixelslic = cv2.ximgproc.createSuperpixelSLIC(merged, algorithm=cv2.ximgproc.MSLIC, region_size=6)

    superpixelslic.getNumberOfSuperpixels()
    superpixelslic.iterate(num_iterations=20)


    mask = superpixelslic.getLabelContourMask()

    # Get the labels and number of superpixels
    labels = superpixelslic.getLabels()
    num_labels = superpixelslic.getNumberOfSuperpixels()

    # Create a mask for each superpixel
    sp_mask = np.zeros(shape=merged.shape[:2])

    merged = merged.astype(float)

    super_pixels = {label: merged[labels == label] for label in range(num_labels)}

    # get the background / vessel
    # background pixels should contain mostly black (0) pixels over dimensions.

    background_pixels = {}
    foreground_pixels = {}

    logger.info("Cluster superpixels into foreground and background pixels")
    for label, pixels in super_pixels.items():
        if pixels[pixels == 0].size / pixels.size > 0:
            background_pixels[label] = pixels
        else:
            foreground_pixels[label] = pixels

    # calculate the mean over each channel within the superpixel
    foreground_pixels_means = {label: np.mean(pixels, axis=0) for label, pixels in foreground_pixels.items()}

    # cluster the superpixels based on the mean channel values within the superpixel
    logger.info(f"Cluster (n={n_clusters}) the foreground superpixels based on superpixel mean values")
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    kmeans.fit(list(foreground_pixels_means.values()))

    # calculate the cluster distance from origin to have a measure for sorting the labels.
    cluster_distances = [np.sqrt(np.dot(np.array(center).T, np.array(center))) for center in kmeans.cluster_centers_]
    sorted_label_idx = np.argsort(cluster_distances)

    # mapping the super pixel key to the kmeans clustering key
    superpixel_kmeans_map = {sp: km for sp, km in zip(foreground_pixels.keys(), kmeans.labels_)}

    # replacing superpixel key with kmeans cluster key
    lookup = np.vectorize(superpixel_kmeans_map.get)
    foreground_clustered = lookup(labels)

    # create template for the background / vessels for classification
    logger.info("Create hierarchical grayscale image of clusters")
    background_template = np.ones_like(merged[:, :, 0]) * 255
    for i in range(n_clusters):
        background_template[foreground_clustered == sorted_label_idx[i]] = 0

    plot_pic(background_template)

    classes, filtered_contours, tissue_boundary = get_and_classify_background_polys(background_template,
                                                                                    foreground_clustered,
                                                                                    sorted_label_idx,
                                                                                    n_clusters)

    logger.info("Save vessel contours with classes")
    with open("vessels.pickle", "wb") as f:
        pickle.dump((classes, filtered_contours), f)
    # shades of gray, n clusters + 2 for background
    template = np.zeros_like(merged[:, :, 0]).astype(np.uint8)
    shades = n_clusters + 2
    for i in range(n_clusters):
        template[foreground_clustered == sorted_label_idx[i]] = (i + 1) * int(255 / shades)

    class_0_contours = [cnt for cnt, class_ in zip(filtered_contours, classes) if class_ == 0]
    class_1_contours = [cnt for cnt, class_ in zip(filtered_contours, classes) if class_ == 1]

    cv2.drawContours(template, class_0_contours, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(template, class_1_contours, -1, 0, thickness=cv2.FILLED)

    template = 255 - template

    tissue_mask = np.zeros_like(template, dtype=np.uint8)
    cv2.drawContours(tissue_mask, [tissue_boundary], -1, 255, thickness=cv2.FILLED)
    tissue_mask = tissue_mask.astype(bool)

    template[~tissue_mask] = 0
    cv2.drawContours(template, [tissue_boundary], -1, 255, thickness=2)

    plot_pic(template)

    logger.info("Run thinning algorithm.")
    thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

    cv2.drawContours(thinned, class_0_contours, -1, 0, thickness=cv2.FILLED)
    cv2.drawContours(thinned, class_0_contours, -1, 255, thickness=1)

    cv2.drawContours(thinned, class_1_contours, -1, 0, thickness=cv2.FILLED)
    cv2.drawContours(thinned, class_1_contours, -1, 255, thickness=1, )

    thinned = cv2.ximgproc.thinning(thinned.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

    plot_pic(thinned)
    cv2.imwrite("thinned.png", thinned)
