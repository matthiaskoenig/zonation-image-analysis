"""Read CZI files.

Look into alternatives for reading images: https://pypi.org/project/pylibCZIrw/
https://github.com/sebi06/czitools

https://allencellmodeling.github.io/aicsimageio/

"""
from pathlib import Path
from typing import List, Tuple, Dict
from xml.etree.ElementTree import XML

import numpy as np
import dask.array as da
import zarr
from aicsimageio import AICSImage, readers
from ome_types import OME
from tifffile import TiffFileError, tifffile

from zia.io.core import check_image_path
from zia.console import console


def parse_metadata(ome_dict: dict) -> Dict:
    """Parse metadata from OME dictionary."""

    info = {
        "NominalMagnification": float(
            ome_dict["OME"]["Instrument"]["Objective"]["@NominalMagnification"]
        ),
        "AquisitionDate": ome_dict["OME"]["Image"]["AcquisitionDate"],
        "SizeX": ome_dict["OME"]["Image"]["Pixels"]["@SizeX"],
        "SizeY": ome_dict["OME"]["Image"]["Pixels"]["@SizeY"],
        "PhysicalSizeX": float(
            ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeX"]
        ),
        "PhysicalSizeXUnit": ome_dict["OME"]["Image"]["Pixels"][
            "@PhysicalSizeXUnit"
        ],
        "PhysicalSizeY": float(
            ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeY"]
        ),
        "PhysicalSizeYUnit": ome_dict["OME"]["Image"]["Pixels"][
            "@PhysicalSizeYUnit"
        ],
        "Channels": ome_dict["OME"]["Image"]["Pixels"]["Channel"],
    }
    return info


def read_czi(image_path: Path) -> np.ndarray:
    """Read image with tifffile library."""
    check_image_path(image_path)

    img = readers.czi_reader.CziReader(image_path)

    metadata: OME = img.ome_metadata
    ome_dict = metadata.dict()
    console.print(metadata.dict())
    # info = parse_metadata(ome_dict=ome_dict)
    # print(info)


    # info = {
    #     "NominalMagnification": float(
    #         ome_dict["OME"]["Instrument"]["Objective"]["@NominalMagnification"]
    #     ),
    #     "AquisitionDate": ome_dict["OME"]["Image"]["AcquisitionDate"],
    #     "SizeX": ome_dict["OME"]["Image"]["Pixels"]["@SizeX"],
    #     "SizeY": ome_dict["OME"]["Image"]["Pixels"]["@SizeY"],
    #     "PhysicalSizeX": float(
    #         ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeX"]
    #     ),
    #     "PhysicalSizeXUnit": ome_dict["OME"]["Image"]["Pixels"][
    #         "@PhysicalSizeXUnit"
    #     ],
    #     "PhysicalSizeY": float(
    #         ome_dict["OME"]["Image"]["Pixels"]["@PhysicalSizeY"]
    #     ),
    #     "PhysicalSizeYUnit": ome_dict["OME"]["Image"]["Pixels"][
    #         "@PhysicalSizeYUnit"
    #     ],
    #     "Channels": ome_dict["OME"]["Image"]["Pixels"]["Channel"],
    # }

    # czi = CziFile(file_path)
    # dimensions = czi.get_dims_shape()
    # print(dimensions)

    # img = AICSImage(file_path, reconstruct_mosaic=False)  # selects the first scene found

    # Read pyramidal data (1 2 4 8 16 32 64 128)

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

    data = read_czi(example_czi)

    napari_viewer.view_czi_data(
        data=read_czi(example_czi),
        channel_names=["channel 1", "channel 2", "channel 3"]
    )

    # # FIXME: read image metadata
    # napari_viewer.view_czi_data(
    #     data=read_czi(example_czi_axios2),
    #     channel_names=["channel 1", "channel 2", "channel 3"]
    # )
