import shutil

import cv2

from zia.ai.path_utils import DatasetPaths
from zia.annotations.config.project_config import ResourcePaths, read_sample_data_config
from zia.annotations.preprocessing.stain_normalization import normalize_stain

resource_paths = ResourcePaths("roy2020")
dataset_paths = DatasetPaths("roy2020")
stain_separation_config = read_sample_data_config()
for p in (resource_paths.base_path / "datasets").iterdir():

    img_path = p / "img" / "0"
    mask_path = p / "gt" / "0"

    if p.name == "test":
        img_path = p / "val" / "img" / "0"
        mask_path = p / "val" / "gt" / "0"

    dataset_mask_path = dataset_paths.base_path / p.name / "mask"
    dataset_image_path = dataset_paths.base_path / p.name / "image"

    for new_p in [dataset_image_path, dataset_mask_path]:
        new_p.mkdir(exist_ok=True, parents=True)

    for f in mask_path.iterdir():
        to_file = dataset_mask_path / f.name
        shutil.copy(f, to_file)

    for f in img_path.iterdir():
        image = cv2.imread(str(f))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        normalized = normalize_stain(image, stain_separation_config.reference_stain_matrix, stain_separation_config.reference_max_conc)

        bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(dataset_image_path / f.name), bgr)
