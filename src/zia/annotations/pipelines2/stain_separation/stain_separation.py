import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from zia import BASE_PATH
from zia.annotations.pipelines2.stain_separation.wsi_stain_separation import separate_stains
from zia.config import read_config
from zia.log import get_logger

logger = get_logger(__file__)


def get_protein(path: Path) -> str:
    if "he" in path.stem.lower():
        return "HE"
    if "gs" in path.stem.lower():
        return "GS"
    if "cyp1a2" in path.stem.lower():
        return "CYP1A2"
    if "cyp2e1" in path.stem.lower():
        return "CYP2E1"
    if "cyp2d6" in path.stem.lower():
        return "CYP2D6"
    if "cyp3a4" in path.stem.lower():
        return "CYP3A4"


if __name__ == "__main__":

    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_registered = config.image_data_path / "rois_registered"
    out_path = config.image_data_path / "stain_separated"
    out_path.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([f for f in data_dir_registered.iterdir() if f.is_dir()])
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for subject_dir in subject_dirs:
            subject = subject_dir.name
            roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
            for roi_dir in roi_dirs:
                roi = roi_dir.name
                images = sorted([f for f in roi_dir.iterdir() if f.is_file()])
                for image in images:
                    try:
                        separate_stains(image, subject, roi, get_protein(image), out_path, executor=executor, overwrite=False, write_h_stain=False)
                    except Exception as e:
                        traceback.print_exc()
                        continue
