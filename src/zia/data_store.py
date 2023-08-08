"""Zarr image storage."""

from enum import Enum
from typing import List, Optional, Tuple, Dict, Type

import numpy as np
import zarr
from PIL.Image import Image

from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.io.wsi_openslide import read_wsi
from zia.path_utils import ImageInfo


class ZarrGroups(str, Enum):
    LIVER_MASK = "liver_mask"
    DAB_STAIN = "dab_stain"
    H_STAIN = "h_stain"
    E_STAIN = "e_stain"


class DataStore:
    """
    Class manages persisting image data that is produced during
    subsequent analysis in the image analysis pipeline like masks
    """

    _chunksize_rgb = (1024, 1024, 3)

    def __init__(self, image_info: ImageInfo):

        self.image_info = image_info
        self.data = zarr.open_group(self.image_info.zarr_path, mode="a")
        self.image = read_wsi(self.image_info.path)
        self._rois: Optional[List[Roi]] = None

    @property
    def rois(self) -> List[Roi]:
        """
        Lazy loads ROIs. This was implemented like this to allow running the component
        without running the RoiFinderComponent before during development. In this case
        the rois will not have not been registered and thus have to be loaded from the
        file system.
        """
        if not self._rois:
            self._rois = self._load_rois_from_file_system()

        return self._rois

    def _load_rois_from_file_system(self) -> List[Roi]:
        return Roi.load_from_file(self.image_info.roi_path)

    def register_rois(self, rois: Optional[List[Roi]]) -> None:
        """
        initializes rois in data store and saves rois to file system.
        """
        self._rois = rois

        Roi.write_to_geojson(
            rois=rois,
            path=self.image_info.roi_path,
        )

    def create_multilevel_group(
        self, zarr_group: ZarrGroups, roi_no: int, data: dict[int, np.ndarray]
    ):
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

    def create_mask_array(
        self, zarr_group: ZarrGroups, roi_no: int, shape: Tuple
    ) -> zarr.core.Array:
        data_group = self.data.require_group(zarr_group.value)
        roi_group = data_group.require_group(str(roi_no))
        return roi_group.zeros(
            str(0),
            shape=shape,
            chunks=(2 ** 12, 2 ** 12),  # 4096
            dtype=bool,
            overwrite=True,
        )

    def create_pyramid_group(self, zarr_group: ZarrGroups, roi_no: int, shape: Tuple,
                             dtype: Type) -> \
        Dict[int, zarr.Array]:
        """Creates a zarr group for a roi to save an image pyramid
        @param zarr_group: the enum to sepcify the zarr subdirectory
        @param roi_no: the number of the ROI
        @param shape: the shape of the array to create
        """
        data_group = self.data.require_group(zarr_group.value)
        roi_group = data_group.require_group(str(roi_no))

        pyramid_dict: Dict[int, zarr.Array] = {}

        h, w = shape[:2]

        for i in range(8):
            chunk_w, chunk_h = 2 ** 12, 2 ** 12  # starting at 4096 going down to align with tiles
            factor = 2 ** i

            new_h, new_w = int(h / factor), int(w / factor)

            if new_w < chunk_w:
                chunk_w = new_w
            if new_h < chunk_h:
                chunk_h = new_h

            pyramid_dict[i] = roi_group.empty(
                str(i),
                shape=(new_h, new_w) + ((shape[2],) if len(shape) == 3 else ()),
                chunks=(chunk_h, chunk_w) + ((shape[2],) if len(shape) == 3 else ()),
                dtype=dtype,
                overwrite=True,
                synchronizer=zarr.ThreadSynchronizer())

        return pyramid_dict

    def get_array(self, zarr_group: ZarrGroups, roi_no: int,
                  level: PyramidalLevel) -> zarr.Array:
        return self.data.get(f"{zarr_group.value}/{roi_no}/{level.value}")

    def read_roi_from_slide(self, roi: Roi, level: PyramidalLevel) -> Image:
        xs_ref, ys_ref = roi.get_bound(PyramidalLevel.ZERO)
        ref_loc = xs_ref.start, ys_ref.start

        xs, ys = roi.get_bound(level)
        size = (xs.stop - xs.start, ys.stop - ys.start)

        return self.image.read_region(location=ref_loc, level=level, size=size)

    def read_region_from_roi(
        self,
        roi_no: int,
        location: Tuple[int, int],
        level: PyramidalLevel,
        size: Tuple[int, int],
    ) -> Image:
        """
        reads relative to the given ROI from the WSI image.
        @param roi_no: index of the ROI to read from
        @param location: Tuple that specifies the location in the ROI
        @param level: the pyramidal level of the WSI image to read from
        @param size: the size of the region to read
        @return: PIL Image
        """
        roi = self.rois[roi_no]
        xs, ys = roi.get_bound(PyramidalLevel.ZERO)
        loc_x, loc_y = location

        shifted_location = xs.start + loc_x, ys.start + loc_y

        return self.image.read_region(shifted_location, level, size)

    def read_full_roi(self, roi_no: int, level: PyramidalLevel) -> Image:
        """
        Reads the region of a ROI on a specific level of the WSI pyramid.
        @param roi_no: index of the ROI to read
        @param level: The pyramidal level
        @return: PIL Image
        """
        roi = self.rois[roi_no]
        xs_ref, ys_ref = roi.get_bound(PyramidalLevel.ZERO)
        ref_loc = xs_ref.start, ys_ref.start
        xs, ys = roi.get_bound(level)
        size = (xs.stop - xs.start, ys.stop - ys.start)
        return self.image.read_region(ref_loc, level, size)
