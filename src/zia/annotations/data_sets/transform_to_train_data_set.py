from pathlib import Path

import cv2

from zia.annotations.config.project_config import ResourcePaths

dataset_paths = ResourcePaths("sample_data")
target_path = Path(r"D:\training_data\sampledata")

mask_path = target_path / "test" / "mask"
img_path = target_path / "test" / "image"

for p in [mask_path, img_path]:
    p.mkdir(parents=True, exist_ok=True)

for mask_file in dataset_paths.mask_path.iterdir():
    name = mask_file.name
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(str(dataset_paths.image_path / mask_file.name))

    img_downscaled = cv2.pyrDown(img)
    mask_downscaled = cv2.pyrDown(mask)

    cv2.imwrite(str(img_path / name), img_downscaled)
    cv2.imwrite(str(mask_path / name), mask_downscaled)
