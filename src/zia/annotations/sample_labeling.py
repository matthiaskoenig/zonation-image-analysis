from typing import List, Tuple
from zipfile import Path

import cv2
import geojson
import matplotlib
import numpy as np
from shapely import Polygon, to_geojson
from tqdm import tqdm

from zia import BASE_PATH
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.pipeline_components.algorithm.droplet_detection.droplet_detection import extract_features, cluster_droplets_trial, cluster_droplets, \
    convert_polys_to_contours


def convert_to_BGR(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = [int(v * 255) for v in rgb[:3]]
    return (b, g, r)


def write_to_geojson(polygons: List[Polygon], labels: np.ndarray, output_path: Path):
    features = []
    for idx, polygon in enumerate(polygons):
        feature = geojson.Feature(id=idx, geometry=polygon.__geo_interface__, properties={"label": int(labels[idx])})
        features.append(feature)

    feature_collection = geojson.FeatureCollection(features=features)

    # Write the GeoJSON data to a file
    with open(str(output_path), 'w') as output_file:
        geojson.dump(feature_collection, output_file, indent=2)


if __name__ == "__main__":
    path = BASE_PATH / "sample_data"

    result_path = path / "classified"
    polygons_path = path / "polygons"

    polygons_path.mkdir(parents=True, exist_ok=True)
    result_path.mkdir(parents=True, exist_ok=True)

    he_path = path / "he_image"
    n_cl = 2

    features_dict = {}
    polygon_dict = {}

    image_paths = [p for p in he_path.iterdir()]

    for p in tqdm(image_paths, "Loading image and extracting features", unit="image"):
        image = cv2.imread(str(p))

        polygons, features = extract_features(image)
        features_dict[p.stem] = features
        polygon_dict[p.stem] = polygons

    fv = np.vstack([f for f in features_dict.values()])

    # cluster_droplets_trial(fv)

    kmeans = cluster_droplets(fv, 0.9, n_cl)

    label_dict = {}
    split_idx = []
    idx = 0

    for val in polygon_dict.values():
        idx += len(val)
        split_idx.append(idx)

    split_labels = np.split(kmeans.labels_, split_idx[:-1])

    label_dict = {key: split for key, split in zip(polygon_dict.keys(), split_labels)}

    for img_name, polygons in polygon_dict.items():
        image = cv2.imread(str(he_path / f"{img_name}.png"))

        for i in range(n_cl):
            class_polys = [poly for poly, label in zip(polygons, label_dict[img_name]) if label == i]

            contours = convert_polys_to_contours(class_polys)
            color = matplotlib.colormaps["tab10"](i)

            color = convert_to_BGR(color)
            cv2.drawContours(image, contours, -1, color, thickness=2)

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_path / f"{img_name}.png"), image)
        write_to_geojson(polygons, label_dict[img_name], polygons_path / f"{img_name}.geojson")

        #plot_pic(image)
