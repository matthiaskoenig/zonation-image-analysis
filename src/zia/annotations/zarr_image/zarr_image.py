"""Zarr image storage."""

from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import dask.array
import numpy as np
import zarr
from dask.array import from_zarr
from tifffile import imread
from zarr import Group

from zia.annotations.annotation.roi import PyramidalLevel, Roi
from zia.annotations.path_utils import FileManager


class ZarrGroups(Enum):
    LIVER_MASK = "liver_mask"


class LeveledRoi:
    def __init__(self):
        self._images: Dict[int, dask.array.Array] = {}
        self._bounds: Dict[int, Tuple[int, int, int, int]] = {}

    def add_level(
        self, level: int, array: dask.array.Array, bound: Tuple[int, int, int, int]
    ) -> None:
        self._images[level] = array
        self._bounds[level] = bound

    def get_by_level(
        self, level: PyramidalLevel
    ) -> Tuple[dask.array.Array, Tuple[int, int, int, int]]:
        return self._images.get(level), self._bounds.get(level)

    def get_down_sized_level(
        self, level: PyramidalLevel
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if level < 4:
            raise NotImplementedError(
                "Resizing for lower levels than four is not supported" "so far."
            )

        arr, _ = self.get_by_level(PyramidalLevel.FOUR)
        return self._resize_mask(level, arr)

    def _resize_mask(
        self, target_level: PyramidalLevel, arr: dask.array.core.Array
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        factor = 2 ** (PyramidalLevel.FOUR - target_level)
        h, w = arr.shape[0:2]
        new_h, new_w = int(h / factor), int(w / factor)

        bgr_image = cv2.cvtColor(arr.compute(), cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(
            bgr_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        bounds = self._bounds.get(PyramidalLevel.FOUR)
        resized_bounds = tuple([int(b / factor) for b in bounds])

        return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), resized_bounds

    def __iter__(
        self,
    ) -> Iterator[Tuple[int, dask.array.Array, Tuple[int, int, int, int]]]:
        for level in self._images:
            yield level, self._images.get(level), self._bounds.get(level)


class ZarrImage:
    _rois: List[LeveledRoi] = None
    _roi_annos: List[Roi] = None
    image: Dict[int, dask.array.Array]
    _chunksize_rgb = (1024, 1024, 3)

    def __init__(self, image_name, file_manager: FileManager):
        """
        initializes this ZarrImage with a zarr store and the original
        image. The image is not loaded in memory but can be accessed over +
        the image dict, which holds the image levels with a dask array
        that wraps the underlying zarr array.
        """
        self._file_manager = file_manager
        self.name = image_name
        self._initialize()
        self.image = self._initialize_original_image()

    @property
    def rois(self) -> List[LeveledRoi]:
        if not self._rois:
            self._register_rois(None)
        return self._rois

    @property
    def roi_annos(self) -> List[Roi]:
        if not self._roi_annos:
            self._register_rois(None)
        return self._roi_annos

    def iter_rois(self) -> Iterator[Tuple[LeveledRoi, Roi]]:
        for roi, roi_anno in zip(self.rois, self.roi_annos):
            yield roi, roi_anno

    def _initialize(self):
        self.data = zarr.open_group(
            self._file_manager.get_zarr_file(self.name), mode="a"
        )

    def _initialize_original_image(self) -> Dict[int, dask.array.Array]:
        with imread(
            self._file_manager.get_image_path(self.name),
            aszarr=True,
            chunkshape=(1024, 1024, 3),
        ) as store:
            file: Group = zarr.open(store=store, mode="r")

            arrays = [from_zarr(file[key]) for key in file.keys()]

            #### calcualting resolution levels for the arrays in the ndpi file
            levels = [int(np.log2(arrays[0].shape[0] / arr.shape[0])) for arr in arrays]

            return {l: arr for l, arr in zip(levels, arrays)}

    def _register_rois(self, rois: Optional[List[Roi]]) -> None:
        if not rois:
            rois = Roi.load_from_file(
                self._file_manager.get_roi_geojson_paths(self.name)
            )
        leveled_rois = []
        print(len(rois))
        for roi in rois:
            leveled_roi = LeveledRoi()
            for level in self.image.keys():
                xs, ys = roi.get_bound(PyramidalLevel.get_by_numeric_level(level))
                leveled_roi.add_level(
                    level,
                    self.image.get(level)[ys, xs],
                    (xs.start, ys.start, xs.stop, ys.stop),
                )
            leveled_rois.append(leveled_roi)
        self._rois, self._roi_annos = leveled_rois, rois

    def create_multilevel_group(
        self, zarr_group: ZarrGroups, roi_no: int, data: dict[int, np.ndarray]
    ):
        data_group = self.data.require_group(zarr_group.value)
        roi_group = data_group.require_group(str(roi_no))
        for i, arr in data.items():
            roi_group.create_dataset(
                str(i),
                shape=arr.shape,
                data=arr,
                chunks=(1024 * 4, 1024 * 4),
                overwrite=True,
            )

    def get_liver_mask(self, roi_no: int, level: PyramidalLevel) -> np.ndarray:
        if not str(roi_no) in self.data.get(ZarrGroups.LIVER_MASK.value).keys():
            raise FileNotFoundError(
                f"This roi number ({roi_no}) does not"
                f" exist in the group '{ZarrGroups.LIVER_MASK.value}' for"
                f"image '{self.name}'."
            )

        if (
            not str(level.value)
            in self.data.get(f"{ZarrGroups.LIVER_MASK.value}/{roi_no}").keys()
        ):
            next_higher_key, arr = self._get_next_lower_key(
                level, ZarrGroups.LIVER_MASK, roi_no
            )
            return self._resize_mask(
                PyramidalLevel.get_by_numeric_level(int(next_higher_key)), level, arr
            )

        return from_zarr(
            self.data.get(f"{ZarrGroups.LIVER_MASK.value}/{roi_no}/{level.value}")
        )

    def _get_next_lower_key(
        self, level: PyramidalLevel, zarr_group: ZarrGroups, roi_no: int
    ) -> Tuple[str, zarr.Array]:
        next_lower_key = None
        for key in self.data.get(f"{zarr_group.value}/{roi_no}"):
            if int(key) < level:
                next_lower_key = key
                break

        if next_lower_key is not None:
            return next_lower_key, self.data.get(
                f"{zarr_group.value}/{roi_no}/{next_lower_key}"
            )
        else:
            raise Exception(
                f"No higher level found for level '{level}' for '{self.name}'."
            )

    def _resize_mask(
        self,
        source_level: PyramidalLevel,
        target_level: PyramidalLevel,
        arr: zarr.Array,
    ) -> np.ndarray:
        factor = 2 ** (source_level - target_level)
        h, w = arr.shape
        new_h, new_w = int(h / factor), int(w / factor)

        boolean_array = arr[:].astype("uint8") * 255
        single_channel_image = cv2.cvtColor(boolean_array, cv2.COLOR_GRAY2BGR)
        resized_image = cv2.resize(
            single_channel_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        return (resized_image[:, :, 0] > 0).astype(bool)
