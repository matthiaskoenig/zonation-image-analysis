import re
from pathlib import Path
from typing import List, Optional, Dict

import cv2
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths, read_stain_separation_config
from zia.annotations.preprocessing.stain_normalization import normalize_stain


def find_match(image_name: str, mask_paths: List[Path]) -> Optional[Path]:
    # print(mask_paths)
    mask_path = list(filter(lambda p: image_name in p.stem, mask_paths))

    if len(mask_path) == 0:
        return None
    return mask_path[0]


def create_manual_data(image_paths: Dict[str, Path], mask_paths: Dict[str, Path], resource_paths: ResourcePaths) -> None:
    stain_separation_config = read_stain_separation_config()

    for image_name, mask_path in tqdm(mask_paths.items(), desc="Normalizing images and creating masks and polygons", unit="images"):
        image = cv2.imread(str(image_paths[image_name]))

        image = image[-1024:, -1024:]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = normalize_stain(image, stain_separation_config.reference_stain_matrix)

        cv2.imwrite(str(resource_paths.image_path / f"{image_name}_manual.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))




if __name__ == "__main__":
    resource_paths = ResourcePaths("jena_data")

    raw_data_dir = resource_paths.base_path / "raw_data"

    mask_manual_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}_vakuolen manuell\.jpg$')
    mask_auto_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}_vakuolen_HE\.jpg$')
    image_pattern = re.compile(r'^J-\d{2}-\d{4}-\d{2}(?:_HE)?\.jpg$')

    manual_mask_paths = [p for p in raw_data_dir.iterdir() if mask_manual_pattern.match(str(p.name))]
    auto_mask_paths = [p for p in raw_data_dir.iterdir() if mask_auto_pattern.match(str(p.name))]

    image_paths = {p.stem: p for p in raw_data_dir.iterdir() if image_pattern.match(str(p.name))}

    print("bla")
    manual_masks_dict = {img_name: mask_path for img_name, mask_path in
                         {img_name: find_match(img_name, manual_mask_paths) for img_name in image_paths.keys()}.items()
                         if mask_path is not None}

    print("bla")
    auto_masks_dict = {img_name: mask_path for img_name, mask_path in
                       {img_name: find_match(img_name.split("_")[0], auto_mask_paths) for img_name in image_paths.keys()}.items()
                       if mask_path is not None}

    print(manual_masks_dict)
    print(auto_masks_dict)

    assert len(image_paths) == len(manual_masks_dict) + len(auto_masks_dict)

    create_manual_data(image_paths, manual_masks_dict, resource_paths)
