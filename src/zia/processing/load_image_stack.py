from pathlib import Path

import numpy as np
import zarr

from zia.annotations.annotation.util import PyramidalLevel
from zia.data_store import ZarrGroups
from zia.processing.filtering import invert_image


def load_image_stack_from_zarr(roi_group: zarr.Group, level: PyramidalLevel) -> np.ndarray:
    """
    loads the protein dab stains from the zarr group
    @param roi_group: the zarr group of the ROI
    @param level: the resolution level to load
    @return: np array of stacked images
    """

    arrays = {}
    for i, a in roi_group.items():
        if i in ["HE"]:
            continue
        arrays[i] = np.array(a.get(f"{level}"))

    return np.stack(list(arrays.values()), axis=-1)

