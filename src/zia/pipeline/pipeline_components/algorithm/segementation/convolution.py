from typing import List

import numpy as np
import zarr
from shapely import Polygon

from zia import BASE_PATH
from zia.pipeline.annotation import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.data_store import ZarrGroups
from imagecodecs.numcodecs import Jpeg2k
import numcodecs
from sklearn.cluster import KMeans

from zia.pipeline.pipeline_components.algorithm.segementation.filtering import invert_image, filter_img

numcodecs.register_codec(Jpeg2k)
import cv2

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.FOUR


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


def get_and_classify_background_polys(binary: np.ndarray, labels: np.ndarray, sorted_label_idx: np.ndarray):
    # get backgorund contours
    kernel = np.ones((3, 3), np.uint8)
    eroded = binary  # cv2.dilate(binary, kernel)

    contours, hierarchy = cv2.findContours(eroded.reshape(template.shape[0], eroded.shape[1], 1).astype(np.uint8), cv2.RETR_TREE,
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

    classes = []
    to_plot = np.zeros_like(eroded, dtype=np.uint8)

    for cont in filtered_contours:
        # min_h, min_w = np.min(cont, axis=0).flatten()
        # max_h, max_w = np.max(cont, axis=0).flatten()

        # create mask from contour
        ct_template = np.zeros_like(eroded, dtype=np.uint8)
        cv2.drawContours(ct_template, [cont], -1, 255, thickness=cv2.FILLED)

        dilated_vessel_mask = cv2.dilate(ct_template, kernel).astype(bool)
        vessel_mask = ct_template.astype(bool)

        diff_mask = vessel_mask ^ dilated_vessel_mask

        boundary_labels = labels[diff_mask]

        class_0 = boundary_labels[boundary_labels == sorted_label_idx[0]]
        class_1 = boundary_labels[boundary_labels == sorted_label_idx[1]]

        if class_0.size > class_1.size:
            classes.append(0)
            c = 100
        elif class_1.size > class_0.size:
            classes.append(1)
            c = 200
        else:
            classes.append(-1)
            c = 255

        cv2.drawContours(to_plot, [cont], -1, c, 2)

    print(classes)
    # plot_pic(eroded)
    plot_pic(to_plot)
    return classes, filtered_contours, tissue_boundary


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        if i in ["HE"]:
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

    super_pixels = {label: merged[labels == label] for label in range(num_labels)}

    # get the background / vessel
    # background pixels should contain mostly black (0) pixels over dimensions.

    background_pixels = {}
    foreground_pixels = {}

    for label, pixels in super_pixels.items():
        if pixels[pixels == 0].size / pixels.size > 0:
            background_pixels[label] = pixels
        else:
            foreground_pixels[label] = pixels

    print(len(background_pixels))

    # calculate the mean over each channel within the superpixel
    foreground_pixels_means = {label: np.mean(pixels, axis=0) for label, pixels in foreground_pixels.items()}

    # cluster the superpixels based on the mean channel values within the superpixel
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(list(foreground_pixels_means.values()))

    # calculate the cluster distance from origin to have a measure for sorting the labels.
    cluster_distances = [np.sqrt(np.dot(np.array(center).T, np.array(center))) for center in kmeans.cluster_centers_]
    sorted_label_idx = np.argsort(cluster_distances)

    # mapping the super pixel key to the kmeans clustering key
    superpixel_kmeans_map = {sp: km for sp, km in zip(foreground_pixels.keys(), kmeans.labels_)}

    # replacing superpixel key with kmeans cluster key
    lookup = np.vectorize(superpixel_kmeans_map.get)
    foreground_clustered = lookup(labels)

    # plot what we have so far
    template = np.ones_like(merged[:, :, 0]) * 255
    for i in range(2):
        template[foreground_clustered == sorted_label_idx[i]] = 0

    background_polys = get_and_classify_background_polys(template, foreground_clustered, sorted_label_idx)
    exit(0)

    # get pericentral polys

    template = np.zeros_like(merged[:, :, 0])
    template[foreground_clustered == sorted_label_idx[0]] = 255
    periportal_polys = get_polys(template)

    # get periportal polys
    template = np.zeros_like(merged[:, :, 0])
    template[foreground_clustered == sorted_label_idx[1]] = 255
    pericentral_polys = get_polys(template)

    # sort background polys by size

    sorted_background = sorted(background_polys, key=lambda x: x.area)

    image = sorted_background.pop(-1)
    background_poly = sorted_background.pop(-1)

    from PIL import Image, ImageDraw

    image = Image.new("L", (labels.shape[1], labels.shape[0]), "black")
    draw = ImageDraw.Draw(image)

    draw.polygon(list(background_poly.exterior.coords), fill="white")

    for poly in pericentral_polys:
        draw.polygon(list(poly.exterior.coords), fill="black")

    draw.polygon(list(background_poly.exterior.coords), outline="white")

    image = np.array(image).astype(np.uint8)
    plot_pic(image)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.dilate(image, kernel)
    plot_pic(eroded)

    thinned = cv2.ximgproc.thinning(eroded.reshape(eroded.shape[0], eroded.shape[1], 1).astype(np.uint8))

    coords = np.argwhere(thinned == 1)

    plot_pic(thinned)

    exit(0)
    for i, poly in enumerate(pericentral_polys):
        """print(list(poly.exterior.coords))
        class_ = background_class[i]
        c = "blue"
        if class_ == 0:
            c = "red"
        elif class_ == 1:
            c = "green"""

        patch = matplotlib.patches.Polygon(list(poly.exterior.coords), fill=True, facecolor="black", linewidth=2)
        ax.add_patch(patch)
        ax.set_xlim(right=labels.shape[1])
        ax.set_ylim(top=labels.shape[0])

    background_patch = matplotlib.patches.Polygon(list(background_poly.exterior.coords), fill=True, facecolor="black", linewidth=2)
    ax.add_patch(background_patch)

    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.show()

    exit(0)
    """super_pixel_means = {label: np.mean(merged[labels == label], axis=0) for label in range(num_labels)}

    print(super_pixel_means)

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

    dist = cv2.distanceTransform(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8), cv2.DIST_L1, 3)

    dist_output = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)

    plot_pic(dist_output)

    thinned = cv2.ximgproc.thinning(template.reshape(template.shape[0], template.shape[1], 1).astype(np.uint8))

    plot_pic(thinned)

    thinned_central = np.copy(thinned)
    thinned_central[(kmeans_repr == sorted_label_idx[2]) & (thinned == 0)] = 120

    plot_pic(thinned_central)
    thinned_vessel = np.copy(thinned)
    thinned_vessel[kmeans_repr == sorted_label_idx[0]] = 100

    plot_pic(thinned_vessel)

    exit(0)

    eroded = template

    for i in range(10):
        eroded = cv2.erode(eroded, kernel)
        contours, hierarchy = cv2.findContours(eroded.reshape(eroded.shape[0], eroded.shape[1], 1).astype(np.uint8), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        ct_template = np.zeros_like(eroded, dtype=np.uint8)
        cv2.drawContours(ct_template, contours, -1, 255, 2)
        # plot_pic(eroded)
        plot_pic(ct_template)

    # print(superpixel_medians)

    # plot_pic(sp_mask)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=600)
"""
    """for key, ax in zip(arrays, axes.flatten()):
        print(conv[key].shape)
        final = np.ma.masked_where(mask, conv[key])
        ax.imshow(final.data, cmap="binary_r")
        ax: plt.Axes
        ax.set_title(key)

    plt.show()"""
    # watershed
    """for i in range(5):
        template[foreground_clustered == sorted_label_idx[i]] = i + 1

    plot_pic(template)

    marker_layer = np.zeros_like(merged[:, :, 0])
    marker_layer[foreground_clustered == sorted_label_idx[4]] = 255

    contours, hierarchy = cv2.findContours(marker_layer.reshape(marker_layer.shape[0], marker_layer.shape[1], 1).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    markers = np.zeros(shape=(marker_layer.shape[0], marker_layer.shape[1], 1), dtype=np.int32)
    markers.fill(-1)

    for i, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], 0, i + 1, thickness=cv2.FILLED)

    plot_pic(markers)

    image_gray = np.zeros(shape=(template.shape[0], template.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        image_gray[:, :, i] = template

    print(markers.shape, image_gray.shape)
    cv2.watershed(image_gray, markers)

    image_gray[markers.reshape(markers.shape[0], markers.shape[1]) == -1] = [0, 0, 255]

    plot_rgb(image_gray)"""
