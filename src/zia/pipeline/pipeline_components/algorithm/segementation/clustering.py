from pathlib import Path
from typing import List, Tuple

import numcodecs
import numpy as np
from imagecodecs.numcodecs import Jpeg2k
from shapely import Polygon
from sklearn.cluster import KMeans

from zia.log import get_logger

numcodecs.register_codec(Jpeg2k)
import cv2

logger = get_logger(__file__)


def check_if_poly_is_in_any_of_the_lists(poly_to_check: Polygon, lists: List[List[Polygon]]) -> int:
    for i, polygons in enumerate(lists):
        for poly in polygons:
            if poly.contains(poly_to_check):
                return i
    return len(lists)


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

    if len(count_vectors) > 1:
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(count_vectors)

        # metric for sorting these clusters
        sorted_idx = np.argsort(np.sum(kmeans.cluster_centers_[:, :2], axis=1))
        classes = [sorted_idx[label] for label in kmeans.labels_]

    elif len(count_vectors) == 1:
        logger.warning("Only one vessel found. Vessel could not be classified by ")

        mean_c = 1 / n_clusters * sum([c * i for i, c in enumerate(count_vectors[0])])
        reference_mean = 1 / n_clusters * sum([i for i in range(n_clusters)])

        if mean_c > reference_mean:
            class_ = 0
        else:
            class_ = 1

        classes = [class_]
    else:
        classes = []

    return classes, classified_contours, tissue_boundary


def get_factor(x, n) -> float:
    return np.log(x) / np.log(n)


def pad_image(image_stack: np.ndarray, pad: int) -> np.ndarray:
    pad_width = ((pad, pad), (pad, pad), (0, 0))
    return np.pad(image_stack, pad_width, mode="constant", constant_values=0)


def run_skeletize_image(image_stack: np.ndarray, n_clusters=5, pad=10, report_path: Path = None) -> Tuple[
    np.ndarray, Tuple[List[int], list]]:
    image_stack = pad_image(image_stack, pad)

    superpixelslic = cv2.ximgproc.createSuperpixelSLIC(image_stack, algorithm=cv2.ximgproc.MSLIC, region_size=6)
    superpixelslic.iterate(num_iterations=20)

    superpixel_mask = superpixelslic.getLabelContourMask(thick_line=False)
    # Get the labels and number of superpixels
    labels = superpixelslic.getLabels()

    num_labels = superpixelslic.getNumberOfSuperpixels()

    if report_path is not None:
        cv2.imwrite(str(report_path / "superpixels.png"), superpixel_mask)

    merged = image_stack.astype(float)

    super_pixels = {label: merged[labels == label] for label in range(num_labels)}

    # get the background / vessel
    # background pixels should contain mostly black (0) pixels over dimensions.

    background_pixels = {}
    foreground_pixels = {}

    logger.info("Cluster superpixels into foreground and background pixels")
    for label, pixels in super_pixels.items():
        if pixels[pixels == 0].size / pixels.size > 0.1:
            background_pixels[label] = pixels
        else:
            foreground_pixels[label] = pixels

    if report_path is not None:
        out_template = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1], 3)).astype(np.uint8)
        for i in range(num_labels):
            if i in background_pixels.keys():
                out_template[labels == i] = np.array([0, 0, 0])
            else:
                out_template[labels == i] = np.array([255, 255, 255])
        cv2.imwrite(str(report_path / "superpixels_bg_fg.png"), out_template)

    # calculate the mean over each channel within the superpixel
    foreground_pixels_means = {label: np.mean(pixels, axis=0) for label, pixels in foreground_pixels.items()}
    # print(foreground_pixels_means)

    if report_path is not None:
        for k in range(merged.shape[2]):
            out_template = np.zeros(shape=(image_stack.shape[0], image_stack.shape[1])).astype(np.uint8)
            for i, mean in foreground_pixels_means.items():
                out_template[labels == i] = mean[k]
            out_template = out_template / np.max(out_template) * 255
            out_template = out_template.astype(np.uint8)
            out_template = cv2.cvtColor(out_template, cv2.COLOR_GRAY2RGB)
            out_template[np.all(out_template != [0, 0, 0], axis=2) & (superpixel_mask == 255)] = [255, 255, 0]
            # out_template=out_template[180:230, 220:270, :]
            cv2.imwrite(str(report_path / f"superpixel_means_{k}.png"), out_template)

    # cluster the superpixels based on the mean channel values within the superpixel
    logger.info(f"Cluster (n={n_clusters}) the foreground superpixels based on superpixel mean values")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
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

    if report_path is not None:
        out_template = np.ones(shape=(image_stack.shape[0], image_stack.shape[1])).astype(np.uint8)
        for i in range(n_clusters):
            out_template[foreground_clustered == sorted_label_idx[i]] = round((i + 1) * 255 / n_clusters)

        cv2.imwrite(str(report_path / "foreground_clustered.png"), out_template)


    classes, filtered_contours, tissue_boundary = get_and_classify_background_polys(background_template,
                                                                                    foreground_clustered,
                                                                                    sorted_label_idx,
                                                                                    n_clusters)

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

    template = cv2.medianBlur(template, 5)

    if report_path is not None:
        cv2.imwrite(str(report_path / "grayscale.png"), template)

    tissue_mask = np.zeros_like(template, dtype=np.uint8)
    cv2.drawContours(tissue_mask, [tissue_boundary], -1, 255, thickness=cv2.FILLED)
    tissue_mask = tissue_mask.astype(bool)

    template[~tissue_mask] = 0
    cv2.drawContours(template, [tissue_boundary], -1, 255, thickness=1)

    if report_path is not None:
        out_template = np.zeros(shape=(merged.shape[0], merged.shape[1], 4)).astype(np.uint8)
        cv2.drawContours(out_template, class_0_contours, -1, (255, 255, 0, 127), thickness=cv2.FILLED)
        cv2.drawContours(out_template, class_0_contours, -1, (255, 255, 0, 255), thickness=2)

        cv2.drawContours(out_template, class_1_contours, -1, (255, 0, 255, 127), thickness=cv2.FILLED)
        cv2.drawContours(out_template, class_1_contours, -1, (255, 0, 255, 255), thickness=2)

        cv2.drawContours(out_template, [tissue_boundary], -1, (255, 255, 255, 255), thickness=3)

        cv2.imwrite(str(report_path / f"classified_vessels.png"), out_template)
        cv2.imwrite(str(report_path / f"final_clustered_map.png"), template)


    logger.info("Run thinning algorithm.")
    thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))


    ## drawing the vessels on the mask and thinn again to prevent pixel accumulations, the segmentation can't hanlde

    cv2.drawContours(thinned, class_0_contours, -1, 0, thickness=cv2.FILLED)
    cv2.drawContours(thinned, class_0_contours, -1, 255, thickness=1)

    cv2.drawContours(thinned, class_1_contours, -1, 0, thickness=cv2.FILLED)
    cv2.drawContours(thinned, class_1_contours, -1, 255, thickness=1, )

    thinned = cv2.ximgproc.thinning(thinned.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

    if report_path is not None:
        cv2.imwrite(str(report_path / "thinned.png"), thinned)

    return thinned, (classes, filtered_contours)

