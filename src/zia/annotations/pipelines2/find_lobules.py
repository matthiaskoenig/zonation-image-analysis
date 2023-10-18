import traceback
from pathlib import Path

import zarr

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.log import get_logger
from zia.processing.clustering import run_skeletize_image
from zia.processing.filtering import Filter
from zia.processing.get_segments import segment_thinned_image
from zia.processing.load_image_stack import load_image_stack_from_zarr
from zia.processing.process_segment import process_line_segments

logger = get_logger(__file__)

pixel_width = 0.22724690376093626  # got that out of Qpath


def find_lobules_for_subject(subject: str, roi: int, roi_group: zarr.Group, results_path: Path, plot=False, pad=10) -> None:
    logger.info(f"Load images for subject {subject}")
    loaded_level = PyramidalLevel.FIVE
    results_path = results_path / subject / f"{roi}"

    logger.info("Load images as stack")
    image_stack = load_image_stack_from_zarr(roi_group, loaded_level)

    logger.info("Applying filters and preprocessing.")

    image_filter = Filter(image_stack, loaded_level)
    final_level, filtered_image_stack = image_filter.prepare_image()

    logger.info("Run superpixel algorithm.")
    thinned, (vessel_classes, vessel_contours) = run_skeletize_image(filtered_image_stack, n_clusters=3, pad=pad)

    logger.info("Segmenting lines in thinned image.")
    line_segments = segment_thinned_image(thinned)

    logger.info("Creating lobule and vessel polygons from line segments and vessel contours.")
    slide_stats = process_line_segments(line_segments,
                                        vessel_classes,
                                        vessel_contours,
                                        final_level,
                                        pad)
    if plot:
        slide_stats.plot()
    slide_stats.to_geojson(result_dir=results_path)


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_stain_separated = config.image_data_path / "stain_separated"
    out_path = config.image_data_path / "slide_statistics"
    out_path.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])
    for subject_dir in subject_dirs:
        zarr_store = zarr.open(store=subject_dir)

        subject = subject_dir.stem
        stain_1 = zarr_store.get(f"{ZarrGroups.STAIN_1.value}")
        if stain_1 is None:
            logger.error(f"Stain 1 does not exists for subject {subject}.")
            continue
        for key, roi_group in stain_1.groups():
            try:
                logger.info(f"Starting lobule segmentation for subject {subject}, roi: {key}")
                find_lobules_for_subject(subject, key, roi_group, out_path, plot=False)

            except Exception as e:
                logger.error(traceback.print_exc())
                continue