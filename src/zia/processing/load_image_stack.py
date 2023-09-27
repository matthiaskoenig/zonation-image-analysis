from typing import Tuple

import numpy as np
import zarr

from zia.annotations.annotation.util import PyramidalLevel


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


def load_image_stack_from_zarr(roi_group: zarr.Group, level=PyramidalLevel.FIVE) -> np.ndarray:
    """
    loads the protein dab stains from the zarr group
    @param roi_group: the zarr group of the ROI
    @param level: level to load
    @return: np array of stacked images
    """

    arrays = {}
    for i, a in roi_group.items():
        if i in ["HE"]:
            continue
        arrays[i] = np.array(a.get(f"{level.value}"))

    return np.stack(list(arrays.values()), axis=-1)
