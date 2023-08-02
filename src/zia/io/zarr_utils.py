from typing import Dict

import cv2
import numpy as np
from zarr import Array


def write_to_pyramid(template: np.ndarray, pyramid_dict: Dict[int, Array], rs: slice,
                     cs: slice):
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
