import traceback
from pathlib import Path

import cv2
import zarr

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines2.find_lobules import find_lobules_for_subject
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.log import get_logger
from zia.processing.clustering import run_skeletize_image
from zia.processing.filtering import Filter
from zia.processing.get_segments import segment_thinned_image
from zia.processing.load_image_stack import load_image_stack_from_zarr
from zia.processing.process_segment import process_line_segments

logger = get_logger(__file__)


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_stain_separated = config.image_data_path / "stain_separated"
    out_path = config.image_data_path / "slide_statistics"
    out_path.mkdir(parents=True, exist_ok=True)

    sample_subject = "UKJ-19-026_Human"

    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])
    for subject_dir in subject_dirs:
        report_path = None

        zarr_store = zarr.open(store=subject_dir)

        subject = subject_dir.stem

        if subject != "UKJ-19-026_Human":
            continue
        stain_1 = zarr_store.get(f"{ZarrGroups.STAIN_1.value}")
        if stain_1 is None:
            logger.error(f"Stain 1 does not exists for subject {subject}.")
            continue
        for key, roi_group in stain_1.groups():
            report_path = config.reports_path / "lobule_seg" / subject
            report_path.mkdir(exist_ok=True, parents=True)
            try:
                logger.info(f"Starting lobule segmentation for subject {subject}, roi: {key}")
                find_lobules_for_subject(subject, key, roi_group, out_path, plot=False, report_path=report_path)

            except Exception as e:
                logger.error(traceback.print_exc())
                continue
