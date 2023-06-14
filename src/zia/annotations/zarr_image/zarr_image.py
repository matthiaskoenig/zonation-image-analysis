from typing import Tuple, Dict, List, Iterator

import dask.array
import numpy as np
import zarr
from dask.array import from_zarr
from tifffile import imread
from zarr import Group

from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.path_utils.path_util import FileManager


class LeveledRoi:
    def __init__(self):
        self._images: Dict[int, dask.array.Array] = {}
        self._bounds: Dict[int, Tuple[int, int, int, int]] = {}

    def add_level(self, level: int, array: dask.array.Array,
                  bound: Tuple[int, int, int, int]) -> None:
        self._images[level] = array
        self._bounds[level] = bound

    def get_by_level(self, level: PyramidalLevel) -> Tuple[
        dask.array.Array, Tuple[int, int, int, int]]:
        return self._images.get(level), self._bounds.get(level)

    def __iter__(self) -> Iterator[
        Tuple[int, dask.array.Array, Tuple[int, int, int, int]]]:
        for level in self._images:
            yield level, self._images.get(level), self._bounds.get(level)


class ZarrImage:
    rois: List[LeveledRoi]
    roi_annos: List[Roi]
    image: Dict[int, dask.array.Array]
    _chunksize_rgb = (1024, 1024, 3)

    def __init__(self, image_name, rois: List[Roi], file_manager: FileManager):
        self._initialize(file_manager.get_zarr_file(image_name))
        self.name = image_name
        self.image = self._initialize_original_image(file_manager.get_image_path(image_name))
        self.rois, self.roi_annos = self._register_rois(rois)

    def _initialize(self, path):
        self.data = zarr.open_group(path, mode="a")

    def _initialize_original_image(self, ndpi_path) -> Dict[int, dask.array.Array]:
        with imread(ndpi_path, aszarr=True,
                    chunkshape=(1024, 1024, 3)) as store:
            file: Group = zarr.open(store=store, mode="r")

            arrays = [from_zarr(file[key]) for key in
                      file.keys()]

            #### calcualting resolution levels for the arrays in the ndpi file
            levels = [int(np.log2(arrays[0].shape[0] / arr.shape[0])) for arr in arrays]

            return {l: arr for l, arr in zip(levels, arrays)}

    def _register_rois(self, rois: List[Roi]) -> Tuple[List[LeveledRoi], List[Roi]]:
        leveled_rois = []
        for roi in rois:
            leveled_roi = LeveledRoi()
            for level in self.image.keys():
                xs, ys = roi.get_bound(PyramidalLevel.get_by_numeric_level(level))
                leveled_roi.add_level(level, self.image.get(level)[ys, xs],
                                      (xs.start, ys.start, xs.stop, ys.stop))
            leveled_rois.append(leveled_roi)
        return leveled_rois, rois

    def create_multilevel_group(self, name: str, roi_no: int,
                                data: dict[int, np.ndarray]):
        data_group = self.data.require_group(name)
        roi_group = data_group.require_group(str(roi_no))
        for i, arr in data.items():
            roi_group.create_dataset(str(i), shape=arr.shape, data=arr,
                                     chunks=(1024 * 4, 1024 * 4), overwrite=True)

