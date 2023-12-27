from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import zarr

from zia.pipeline.annotation import PyramidalLevel
from zia.log import get_logger

from imagecodecs.numcodecs import Jpeg
from numcodecs import register_codec

register_codec(Jpeg)
log = get_logger(__file__)


def get_level_to_load(roi_group: zarr.Group, max_size: float) -> int:
    protein = roi_group.get("HE")
    # for key in protein.keys():
    # print(key)
    for level, level_array in protein.items():
        print(level)
        if level_array.shape[0] * level_array.shape[1] <= max_size:
            print(level_array.shape)
            return level
    return PyramidalLevel.SEVEN


def load_image_stack_from_zarr(zarr_paths: Dict[str, Path], level=PyramidalLevel.FIVE, throw_out_ratio: float = 0.8) -> np.ndarray:
    """
    loads the protein dab stains from the zarr group
    @param roi_group: the zarr group of the ROI
    @param level: level to load
    @param throw_out_ratio: the min percentage of non_background pixel for the slide to have to not be discarded
    @return: np array of stacked images
    """

    arrays = {}
    for protein, zarr_path in zarr_paths.items():
        if protein in ["HE"]:
            continue
        arrays[protein] = np.array(zarr.open_array(store=zarr_path, path=f"{level.value}"))

    counts = {i: np.count_nonzero(arr != 255) for i, arr in arrays.items()}
    median_count = np.median(list(counts.values()))
    arrays = {i: arr for i, arr in arrays.items() if counts[i] / median_count > throw_out_ratio}
    for i, count in counts.items():
        if i not in arrays:
            ratio = count / median_count
            log.info(f"Discarded {i} with non background pixel ratio : {ratio:.2f}")

    return np.stack(list(arrays.values()), axis=-1)
