import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from zia.annotations.config.project_config import read_sample_data_config, ResourcePaths, SampleDataConfig
from zia.annotations.preprocessing.polygon_classification import extract_features, create_predictions
from zia.annotations.preprocessing.stain_normalization import normalize_stain
from zia.io.wsi_tifffile import read_ndpi
from zia.log import get_logger
from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.file_management.file_management import SlideFileManager
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent

logger = get_logger(__file__)


def create_data(image: zarr.Array, position: Tuple[int, int], shape: Tuple[int, int],
                stain_separation_config: SampleDataConfig) -> np.ndarray:
    h, w = position
    height, width = shape
    sub_array = image[h: h + height, w: w + width]

    normalized = normalize_stain(sub_array, stain_separation_config.reference_stain_matrix, stain_separation_config.reference_max_conc)

    bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)

    return bgr


def create_sample_data():
    sample_data_config = read_sample_data_config()
    resource_paths = ResourcePaths("sample_data")

    sample_data = []

    features_dict = {}
    polygon_dict = {}

    for group in ["steatosis", "control"]:
        project_config = get_project_config(group)
        file_manager = SlideFileManager(project_config.data_path, project_config.extension)
        for subject, slides in tqdm(file_manager.group_by_subject().items(), desc=f"Creating sample dataset for {group}", unit="slides"):

            slide = None
            for sl in slides:
                if sl.protein == "he":
                    slide = sl
                    break

            image = read_ndpi(project_config.data_path / slide.species / slide.protein.upper() / f"{slide.name}.{project_config.extension}")[0]

            point_path = sample_data_config.tile_coords_path / group / f"{slide.name}.ndpi-points.tsv"

            if not point_path.exists():
                continue

            points = pd.read_csv(point_path, sep="\t", index_col=False)

            for i, row in points.iterrows():
                x, y = int(row["x"]), int(row["y"])
                image_name = f"{subject}_{i}.png"

                sample_data.append(
                    {
                        "image": image_name,
                        "dataset": "sample_data",
                        "subject": slide.subject,
                        "species": slide.species,
                        "group": "steatosis",
                        "sample": i,
                        "position": [x, y],
                    }
                )

                bgr = create_data(image, (y, x), (1024, 1024), sample_data_config)
                cv2.imwrite(str(resource_paths.image_path / image_name), bgr)

                mask, polygons, feature_vectors = extract_features(bgr)
                cv2.imwrite(str(resource_paths.raw_masks / image_name), mask)

                if feature_vectors is not None:
                    features_dict[f"{subject}_{i}"] = feature_vectors
                    polygon_dict[f"{subject}_{i}"] = polygons

    with open(resource_paths.base_path / "data_config.json", "w") as outfile:
        json.dump(sample_data, outfile)

    logger.info("Classifying and saving polygons.")
    create_predictions(features_dict, polygon_dict, resource_paths)


if __name__ == "__main__":
    create_sample_data()
