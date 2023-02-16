"""Read CZI files.

Look into alternatives for reading images: https://pypi.org/project/pylibCZIrw/
https://github.com/sebi06/czitools

https://allencellmodeling.github.io/aicsimageio/

"""
from pathlib import Path
from typing import List, Tuple
from xml.etree.ElementTree import XML

import numpy as np
import dask.array as da
import zarr
from aicsimageio import AICSImage, readers
from tifffile import TiffFileError, tifffile

from zia.io.core import check_image_path


def read_czi(image_path: Path) -> np.ndarray:
    """Read image with tifffile library."""
    check_image_path(image_path)

    img = readers.czi_reader.CziReader(image_path)

    # czi = CziFile(file_path)
    # dimensions = czi.get_dims_shape()
    # print(dimensions)

    # img = AICSImage(file_path, reconstruct_mosaic=False)  # selects the first scene found

    properties = [
        img.dims,  # returns a Dimensions object
        img.dims.order,  # returns string "TCZYX"
        img.dims.X,  # returns size of X dimension
        img.shape,  # returns tuple of dimension sizes in TCZYX order
        img.scenes,  # returns total number of pixels
    ]

    print(properties)
    # print(img.get_image_data("TCZYXS"))

    # data: np.ndarray = img.get_image_data("TCZYXS")
    data: np.ndarray = img.get_image_data("XYC")
    print(type(data))
    print(data.shape)  # (1, 3, 1, 1040, 1388, 1)
    # handle pyramidal information correctly


    return data

    # raise e


if __name__ == "__main__":
    import dask.array as da

    from zia import napari_viewer
    from zia import (
        example_czi,
        example_czi_axios1,
        example_czi_axios2,
        example_czi_axios3,
        example_czi_axios4,
    )


    data: List[da.Array] = read_czi(example_czi_axios2)

    # FIXME: display channels
    napari_viewer.view_ndpi_data(data=data)
