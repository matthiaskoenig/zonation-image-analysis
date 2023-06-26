"""Zarr image storage."""

from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import dask.array
import numpy as np
import zarr

from zia.io.wsi_openslide import openslide, read_wsi
from dask.array import from_zarr
from tifffile import imread
from zarr import Group

from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.path_utils import FileManager


class ZarrGroups(Enum):
    LIVER_MASK = "liver_mask"
    DAB_STAIN = "dab_stain"


class DataStore:
    """
    Class manages persisting image data that is produces during
    subsequent analysis in the image analysis pipeline like masks
    """

    _chunksize_rgb = (1024, 1024, 3)

    def __init__(self, image_name, file_manager: FileManager):
        self._rois: Optional[List[Roi]] = None  # lazy initialization
        self._file_manager = file_manager
        self.name = image_name
        self.data = zarr.open_group(self._file_manager.get_zarr_file(self.name), mode="a")
        self.image = read_wsi(self._file_manager.get_image_path(self.name))

    @property
    def rois(self) -> List[Roi]:
        if not self._rois:
            self._register_rois(None)
        return self._rois

    def _register_rois(self, rois: Optional[List[Roi]]) -> None:
        if not rois:
            self._rois = Roi.load_from_file(self._file_manager.get_roi_geojson_paths(self.name))
        else:
            self._rois = rois

    def create_multilevel_group(self, zarr_group: ZarrGroups, roi_no: int, data: dict[int, np.ndarray]):
        data_group = self.data.require_group(zarr_group.value)
        roi_group = data_group.require_group(str(roi_no))
        for i, arr in data.items():
            roi_group.creaXte_dataset(
                str(i),
                shape=arr.shape,
                data=arr,
                chunks=(1024 * 4, 1024 * 4),
                overwrite=True,
            )

    def create_mask_array(self, zarr_group: ZarrGroups, roi_no: int, shape: Tuple) -> zarr.core.Array:
        data_group = self.data.require_group(zarr_group.value)
        roi_group = data_group.require_group(str(roi_no))
        return roi_group.zeros(
            str(0),
            shape=shape,
            chunks=(1024 * 4, 1024 * 4),
            dtype=bool,
            overwrite=True,
        )
