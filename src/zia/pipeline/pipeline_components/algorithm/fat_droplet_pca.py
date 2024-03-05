from itertools import compress
from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np
import shapely
import skimage
import zarr
from matplotlib import pyplot as plt
from shapely import Polygon, minimum_bounding_radius, minimum_clearance, minimum_rotated_rectangle
from skimage.feature import peak_local_max
from skimage import measure

from zia.io.wsi_tifffile import read_ndpi
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.slicing import get_tile_slices

import scipy.ndimage as ndi

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def detect_droplets_on_tile(tile: np.ndarray) -> List[Polygon]:
    gs = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)

    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # finding the local maxima of the distance transform and create markers for water shedding
    plm = peak_local_max(dist_transform, min_distance=10)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(plm.T)] = True
    markers, _ = ndi.label(mask)

    segmented = skimage.segmentation.watershed(255 - dist_transform, markers, mask=opening)
    segmented = segmented - 1
    contours, _ = cv2.findContours(segmented, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    polys = [Polygon(np.squeeze(cnt)) for cnt in contours]

    circular, non_circular = filter_solidity(polys)

    return circular


def solidity(polygon: Polygon) -> float:
    return polygon.area / polygon.convex_hull.area


def circularity(polygon: Polygon) -> float:
    return (4 * np.pi * polygon.area) / polygon.length ** 2


def roundness(polygon: Polygon) -> float:
    return (4 * np.pi * polygon.area) / (polygon.convex_hull.length) ** 2


def convexity(polygon: Polygon) -> float:
    return polygon.convex_hull.length / polygon.exterior.length


def sphericity(polygon: Polygon) -> float:
    pcoords = np.vstack(polygon.exterior.xy).T
    centroid = [polygon.centroid.x, polygon.centroid.y]

    r_inner = np.min(np.linalg.norm(pcoords - centroid, axis=1))
    r_outer = minimum_bounding_radius(polygon)
    return r_inner / r_outer


def bend_ratio(polygon: Polygon) -> float:
    angles = []
    for k in range(1, len(polygon.exterior.coords) - 1):
        p0 = np.array(polygon.exterior.coords[k - 1])
        p1 = np.array(polygon.exterior.coords[k])
        p2 = np.array(polygon.exterior.coords[k + 1])
        d1 = p0 - p1
        d2 = p2 - p1

        cosine_angle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))

        if cosine_angle >= 0:
            angles.append(True)
        else:
            angles.append(False)

    return np.count_nonzero(angles) / len(angles)


def elongation(polygon: Polygon) -> float:
    bounding_rect = minimum_rotated_rectangle(polygon)

    min_x, min_y, max_x, max_y = bounding_rect.bounds

    width = max_x - min_x
    height = max_y - min_y

    if height > width:
        return width / height

    return height / width


def compactness(polygon: Polygon) -> float:
    return 4 * np.pi * polygon.area / polygon.length ** 2


def feature_vector(p: Polygon, feature_funs: List[Callable[[Polygon], float]]) -> List[float]:
    return [fun(p) for fun in feature_funs]


def clean_droplets(polygons: List[Polygon]) -> Tuple[List[Polygon], List[Polygon]]:
    polygons = [shapely.simplify(p, 3) for p in polygons]
    circularity_filtered = list(filter(lambda p: circularity(p) < 1.5, polygons))

    return circularity_filtered, [p for p in polygons if p not in circularity_filtered]


def filter_circularity(polygons: List[Polygon]) -> List[Polygon]:
    return list(filter(lambda p: circularity(p) < 3, polygons))


def detect_droplets(array: zarr.Array):
    slices = get_tile_slices(array.shape[:2], (2 ** 12, 2 ** 12), pad=200)

    for cs, rs in slices:
        tile = array[cs, rs]
        if np.all(tile == 0):
            continue


def filter_solidity(polygons: List[Polygon]) -> Tuple[List[Polygon], List[Polygon]]:
    _solid = list(filter(lambda p: p.area / p.convex_hull.area > 0.9, polygons))
    _non_solid = list(filter(lambda p: p not in _solid, polygons))
    return _solid, _non_solid


def filter_size(polygons: List[Polygon], min_diameter=6, pixel_size=0.22) -> List[Polygon]:
    return list(filter(lambda p: p.area >= np.pi * (min_diameter / pixel_size), polygons))


if __name__ == "__main__":
    path = Path("D:/image_data/steatosis/RoiExtraction/FLR-180/0/J-15-0789_FLR-180_Lewis_HE_Run 06_LLL02_MAA_003.ome.tiff")
    array = read_ndpi(path)[0]

    print(array.shape)

    sub_array = array[8750 * 2: 8750 * 2 + 2048, 12500 * 2: 12500 * 2 + 2048]
    plot_pic(sub_array)

    gs = cv2.cvtColor(sub_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plot_pic(gs)

    plt.hist(gs.ravel(), 256, [0, 256])
    plt.show()

    plt.hist(gs.ravel(), 256, [0, 256])
    plt.show()
    # plot_pic(gs)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)

    # bilateral_filter = cv2.GaussianBlur(gs, (11, 11), sigmaX=5, sigmaY=5)
    plot_pic(bilateral_filter)

    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 200, 255, cv2.THRESH_BINARY)
    print(_)
    # plot_pic(thresholded)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    plot_pic(opening, "cleaned up mask")

    ## get the contours from the opened binary mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # plot_pic(sub_array)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # plot_pic(dist_transform, "distance transform")

    # finding the local maxima of the distance transform and create markers for water shedding
    plm = peak_local_max(dist_transform, min_distance=10)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(plm.T)] = True
    # plot_pic(mask, "peak local max")
    markers, _ = ndi.label(mask)

    segmented = skimage.segmentation.watershed(255 - dist_transform, markers, mask=opening)

    segmented = segmented - 1

    contours, _ = cv2.findContours(segmented, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = [Polygon(np.squeeze(cnt)) for cnt in contours]
    polygons = filter_size(polygons)

    feature_funs = [circularity, roundness, convexity, sphericity, elongation, compactness, solidity]
    feature_vectors = np.vstack([feature_vector(p, feature_funs) for p in polygons])

    # print(feature_vectors)
    # integrate_droplets(polygons)

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(feature_vectors)

    print(feature_vectors)

    pca = PCA()

    pca.fit(scaled_features)

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(feature_vectors.shape[1]), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle="--")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative Explained Variance")
    plt.show()

    pca = PCA(n_components=3)

    pca.fit(scaled_features)

    scores_pca = pca.transform(scaled_features)

    wcss = []
    for i in range(1, 21):
        kmeans_pca = KMeans(n_clusters=i, n_init="auto")
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)

    fig, ax = plt.subplots(dpi=300)
    ax.plot(range(1, 21), wcss, marker='o', linestyle="--")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("K-means with PCA Clustering")
    plt.show()

    kmeans_pca = KMeans(n_clusters=2, n_init="auto")

    kmeans_pca.fit(scores_pca)

    label_0 = scores_pca[kmeans_pca.labels_ == 0]
    label_1 = scores_pca[kmeans_pca.labels_ == 1]

    fig, ax = plt.subplots(dpi=300)
    ax: plt.Axes

    ax.scatter(label_0[:, 0], label_0[:, 2], c="blue", alpha=0.2)
    ax.scatter(label_1[:, 0], label_1[:, 2], c="red", alpha=0.2)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.show()

    polys_0 = list(compress(polygons, kmeans_pca.labels_ == 0))
    polys_1 = list(filter(lambda p: p not in polys_0, polygons))

    # circular, non_circular = filter_solidity(polygons)

    circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in polys_0]
    non_circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in polys_1]

    cv2.drawContours(sub_array, circular_cnts, -1, (0, 255, 0), 2)
    cv2.drawContours(sub_array, non_circular_cnts, -1, (255, 0, 0), 2)

    plot_pic(sub_array, "first watershed, then clustering")
