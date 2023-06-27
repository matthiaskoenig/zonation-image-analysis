"""Zarr image storage."""

from enum import Enum
from typing import List, Optional, Tuple

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

        # FIXME: load the ROIS at the beginning! I.e. all the data and ROIs
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
        same as read region, but offsets location to roi
        size is absolute. So the size for the level must be calculated beforehand.
        """
        roi = self.rois[roi_no]
        xs, ys = roi.get_bound(PyramidalLevel.ZERO)
        loc_x, loc_y = location

        shifted_location = xs.start + loc_x, ys.start + loc_y

        return self.image.read_region(shifted_location, level, size)
