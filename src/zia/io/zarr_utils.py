from typing import Dict, List, Tuple

import cv2
import numpy as np
import zarr.hierarchy
from zarr import Array

from zia.annotations.annotation.util import PyramidalLevel
from zia.data_store import ZarrGroups



def create_pyramid(template: np.ndarray) -> List[np.ndarray]:
    image_pyramid = [template]

    for i in range(7):
        down_sampled = cv2.pyrDown(image_pyramid[i])

        if (len(template.shape) == 3):
            down_sampled = down_sampled.reshape(down_sampled.shape[0],
                                                down_sampled.shape[1],
                                                template.shape[2])

        image_pyramid.append(down_sampled)

    return image_pyramid


def write_slice_to_zarr_location(slice_image: np.ndarray,
                                 image_pyramid: Dict[int, str],
                                 tile_slices: Tuple[slice, slice],
                                 zarr_store_address: str,
                                 synchronizer: zarr.sync.ThreadSynchronizer):
    rs, cs = tile_slices
    slice_pyramid = create_pyramid(slice_image)
    # persist tile pyramidacally
    for i, tile_image in enumerate(slice_pyramid):
        # print(zarr_store_address, image_pyramid[i])
        zarr_array = zarr.convenience.open_array(store=zarr_store_address,
                                                 path=image_pyramid[i],
                                                 synchronizer=synchronizer)

        factor = 2 ** i

        # resize the slice for level
        new_rs = slice(int(np.ceil(rs.start / factor)), int(np.ceil(rs.stop / factor)))
        new_cs = slice(int(np.ceil(cs.start / factor)), int(np.ceil(cs.stop / factor)))

        zarr_array[new_rs, new_cs] = tile_image.astype(zarr_array.dtype)


def write_to_pyramid(template: np.ndarray, pyramid_dict: Dict[int, Array], rs: slice, cs: slice):
    image_pyramid = {0: template}

    for i in range(len(pyramid_dict) - 1):
        down_sampled = cv2.pyrDown(image_pyramid[i])

        if (len(template.shape) == 3):
            down_sampled = down_sampled.reshape(down_sampled.shape[0],
                                                down_sampled.shape[1],
                                                template.shape[2])

        image_pyramid[i + 1] = down_sampled

    # persist tile pyramidacilly
    for i, level_array in pyramid_dict.items():
        factor = 2 ** i

        # resize the slice for level
        new_rs = slice(int(rs.start / factor), int(rs.stop / factor))
        new_cs = slice(int(cs.start / factor), int(cs.stop / factor))

        level_array[new_rs, new_cs] = image_pyramid[i].astype(level_array.dtype)
