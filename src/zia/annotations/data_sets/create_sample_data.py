import json
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths, read_stain_separation_config
from zia.annotations.preprocessing.polygon_classification import extract_features, classify_and_save_polygons
from zia.annotations.preprocessing.stain_normalization import normalize_stain
from zia.io.wsi_tifffile import read_ndpi
from zia.log import get_logger
from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent

logger = get_logger(__file__)


def get_he_image_path(p: Path) -> Path:
    for p in p.iterdir():
        if p.is_file() and "HE" in p.stem:
            return p
    raise FileNotFoundError(f"No HE image found in {p}")


def create_data(subject: str, roi: str, positions: List[Tuple[int, int]], group: str, shape: Tuple[int, int], resource_paths: ResourcePaths) -> List[
    np.ndarray]:
    project_config = get_project_config(group)
    stain_separation_config = read_stain_separation_config()
    path = project_config.image_data_path / RoiExtractionComponent.dir_name / subject / roi
    he_path = get_he_image_path(path)

    array = read_ndpi(he_path)[0]

    images = []

    height, width = shape

    for i, position in enumerate(positions):
        h, w = position
        sub_array = array[h: h + height, w: w + width]

        normalized = normalize_stain(sub_array, stain_separation_config.reference_stain_matrix, stain_separation_config.reference_max_conc)

        images.append(normalized)

        bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(resource_paths.image_path / f"{subject}_{roi}_{group}_{i}.png"), bgr)

    return images


def create_sample_dataset():
    resource_path = ResourcePaths("sample_data")

    with open(resource_path.base_path / "data_config.json", "r") as f:
        sample_data = json.load(f)

    features_dict = {}
    polygon_dict = {}

    h = sample_data["config"]["height"]
    w = sample_data["config"]["width"]

    for entry in tqdm(sample_data["data"], desc="Normalizing images and extracting features", unit="image"):
        subject = entry["subject"]
        roi = entry["roi"]
        group = entry["group"]
        positions = entry["positions"]

        images = create_data(subject, roi, positions, group, (h, w), resource_path)

        for i, image in enumerate(images):
            mask, polygons, feature_vectors = extract_features(image)
            cv2.imwrite(str(resource_path.mask_path / f"{subject}_{roi}_{group}_{i}.png"), mask)

            if feature_vectors is not None:
                features_dict[f"{subject}_{roi}_{group}_{i}"] = feature_vectors
                polygon_dict[f"{subject}_{roi}_{group}_{i}"] = polygons

    logger.info("Classifying and saving polygons.")
    classify_and_save_polygons(features_dict, polygon_dict, resource_path)


if __name__ == "__main__":
    create_sample_dataset()
