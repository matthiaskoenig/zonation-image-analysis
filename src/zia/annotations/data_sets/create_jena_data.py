import re
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
from shapely import Polygon
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths, read_sample_data_config
from zia.annotations.preprocessing.polygon_classification import write_to_geojson
from zia.annotations.preprocessing.stain_normalization import normalize_stain


def find_match(image_name: str, mask_paths: List[Path]) -> Optional[Path]:
    # print(mask_paths)
    mask_path = list(filter(lambda p: image_name in p.stem, mask_paths))

    if len(mask_path) == 0:
        return None
    return mask_path[0]


def create_mask(mask_image: np.ndarray) -> np.ndarray:
    mask_image = mask_image[-1024:, -1024:]

    blue = mask_image[:, :, 0]
    green = mask_image[:, :, 1]
    red = mask_image[:, :, 2]

    return (((green < 30) | (red < 30)) & (blue > 150)).astype(np.uint8) * 255


def extract_polygons_from_mask(mask: np.ndarray, size_cutoff: int) -> List[Polygon]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > size_cutoff]

    polygons = [Polygon(np.squeeze(cnt)) for cnt in contours if len(cnt) >= 3]
    polygons = [p.simplify(1) for p in polygons]

    return polygons


def create_data(image_paths: Dict[str, Path], mask_paths: Dict[str, Path], mode: str, resource_paths: ResourcePaths) -> None:
    stain_separation_config = read_sample_data_config()

    for image_name, mask_path in tqdm(mask_paths.items(), desc="Normalizing images and creating masks and polygons", unit="images"):
        image = cv2.imread(str(image_paths[image_name]))

        image = image[-1024:, -1024:]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = normalize_stain(image, stain_separation_config.reference_stain_matrix, stain_separation_config.reference_max_conc)

        cv2.imwrite(str(resource_paths.image_path / f"{image_name}_{mode}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        mask_image = cv2.imread(str(mask_path))

        mask = create_mask(mask_image)

        cv2.imwrite(str(resource_paths.mask_path / f"{image_name}_{mode}.png"), mask.astype(np.uint8))

        polygons = extract_polygons_from_mask(mask, size_cutoff=350)

        write_to_geojson(polygons, dict(label="makrosteatosis"), resource_paths.polygon_path / f"{image_name}_{mode}.geojson")


def create_jena_dataset() -> None:
    resource_paths = ResourcePaths("jena_data")

    raw_data_dir = resource_paths.base_path / "raw_data"

    mask_manual_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}_vakuolen manuell\.jpg$')
    mask_auto_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}_vakuolen_HE\.jpg$')
    image_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}(?:_HE)?\.jpg$')

    manual_mask_paths = [p for p in raw_data_dir.iterdir() if mask_manual_pattern.match(str(p.name))]
    auto_mask_paths = [p for p in raw_data_dir.iterdir() if mask_auto_pattern.match(str(p.name))]

    image_paths = {p.stem: p for p in raw_data_dir.iterdir() if image_pattern.match(str(p.name))}

    manual_masks_dict = {img_name: mask_path for img_name, mask_path in
                         {img_name: find_match(img_name, manual_mask_paths) for img_name in image_paths.keys()}.items()
                         if mask_path is not None}

    auto_masks_dict = {img_name: mask_path for img_name, mask_path in
                       {img_name: find_match(img_name.split("_")[0], auto_mask_paths) for img_name in image_paths.keys()}.items()
                       if mask_path is not None}

    assert len(image_paths) == len(manual_masks_dict) + len(auto_masks_dict)

    for mask_dict, mode in zip([manual_masks_dict, auto_masks_dict], ["manual", "auto"]):
        create_data(image_paths, mask_dict, mode, resource_paths)


if __name__ == "__main__":
    create_jena_dataset()
