import multiprocessing
import queue
import threading
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryFile
import time
from tempfile import TemporaryFile
from typing import Tuple, List, Any, IO

import cv2
import dask
import numpy as np
import zarr

from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import \
    calculate_stain_matrix, \
    deconvolve_image, find_max_c
from zia.data_store import DataStore, ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_to_pyramid
from zia.log import get_logger

logger = get_logger(__name__)

def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]
    return full_image_array


def separate_stains(data_store: DataStore, p=0.01, level=PyramidalLevel.ZERO) -> None:
    arrays = read_ndpi(data_store.image_info.path)
    full_image_array: zarr.Array = arrays[level]

    pool = Pool(multiprocessing.cpu_count()-1)

    for roi_no, roi in enumerate(data_store.rois):
        logger.info(f"Start stain separation for ROI {roi_no}")

        mask = data_store.get_array(ZarrGroups.LIVER_MASK, roi_no, level)

        roi_cs, roi_rs = roi.get_bound(level)

        roi_h, roi_w = roi_rs.stop - roi_rs.start, roi_cs.stop - roi_cs.start

        zarr_group = ZarrGroups.E_STAIN if data_store.image_info.metadata.protein == "he" else ZarrGroups.DAB_STAIN
        # create pyramidal group to persist mask
        pyramid_dict_dab = data_store.create_pyramid_group(
            zarr_group, roi_no, (roi_h, roi_w), np.uint8
        )

        pyramid_dict_he = data_store.create_pyramid_group(
            ZarrGroups.H_STAIN, roi_no, (roi_h, roi_w), np.uint8
        )

        slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=2 ** 12)
        final_slices = get_final_slices((roi_rs, roi_cs), slices)

        # compute otsu threshold
        samples = []

        t_s = time.time()

        samples = pool.map(partial(calculate_samples_for_otsu, image_path=data_store.image_info.path, p=p), final_slices)

        #for final_slice in slices:
         #   samples.append(calculate_samples_for_otsu(final_slice, full_image_array, p))

        samples = np.hstack(samples).astype(np.uint8)

        th, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        t_e = time.time()

        logger.info(f"Calculate Otsu threshold {th} in {(t_e - t_s) / 60} min.")

        # calculate stain matrix and c max
        t_s = time.time()

        pxi_samples = []
        temp_files = []

        for i, (tile_slice, final_slice) in enumerate(zip(slices, final_slices)):
            px_sample, temp_file = calculate_stain_matrix_and_cmax(final_slice, tile_slice, full_image_array, mask, th, p)
            pxi_samples.append(px_sample)
            temp_files.append(temp_file)

        pxi_samples = np.vstack(pxi_samples)

        stain_matrix = calculate_stain_matrix(pxi_samples)
        maxC = find_max_c(pxi_samples, stain_matrix)

        print(stain_matrix, maxC)

        t_e = time.time()

        print(
            f"Calculated stain matrix and max concentrations in {(t_e - t_s) / 60} min.")

        # deconvolute image and save
        t_s = time.time()
        for i, (rs, cs) in enumerate(slices):
            tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

            final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
            final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

            # read image tile from the open slide

            image_tile = full_image_array[final_rs, final_cs]

            temp_files[i].seek(0)
            idx = np.load(temp_files[i], allow_pickle=True)
            temp_files[i].close()

            px_in_mask = image_tile[idx]

            template_dab = np.ones(shape=tile_shape).astype(
                np.uint8) * 255
            template_h = np.ones(shape=tile_shape).astype(np.uint8) * 255

            hematoxylin, dab = deconvolve_image(px_in_mask, stain_matrix, maxC)

            template_dab[idx] = dab
            template_h[idx] = hematoxylin

            # print(template.shape)

            write_to_pyramid(template_dab, pyramid_dict_dab, rs, cs)
            write_to_pyramid(template_h, pyramid_dict_he, rs, cs)

            # plot_pic(template)
        t_e = time.time()

        logger.info(f"Deconvoluted ROI image in {(t_e - t_s) / 60} min")


def calculate_stain_matrix_and_cmax(final_slices: Tuple[slice, slice],
                                    tile_slices: Tuple[slice, slice],
                                    image_array: zarr.Array,
                                    mask: zarr.Array,
                                    th: int,
                                    p) -> Tuple[np.ndarray, IO]:
    rs, cs = tile_slices

    final_rs, final_cs = final_slices

    tile_mask = mask[rs, cs]
    # read image tile from the open slide

    image_tile = image_array[final_rs, final_cs]

    # pixels in mask and otsu filtered

    idx = tile_mask & (np.dot(image_tile, np.array([0.587,
                                                    0.114,
                                                    0.299])) < th)

    temp_file = TemporaryFile()
    np.save(temp_file, idx, allow_pickle=True)

    px_in_mask = image_tile[idx]

    choice = np.random.choice(a=[True, False], size=len(px_in_mask),
                              p=[p, 1 - p])

    px_sample = px_in_mask[choice]

    return px_sample, temp_file


def calculate_samples_for_otsu(final_slice: Tuple[slice, slice],
                               image_path: Path,
                               p):
    final_rs, final_cs = final_slice

    tile_shape = (final_rs.stop - final_rs.start, final_cs.stop - final_cs.start)
    image_array = get_toplevel_array(image_path)
    image_tile = image_array[final_rs, final_cs]

    choice = np.random.choice(a=[True, False],
                              size=tile_shape,
                              p=[p, 1 - p])

    sample = image_tile[choice]
    sample_gs = sample.dot(np.array([0.587, 0.114, 0.299]))
    return sample_gs


def get_final_slices(roi_slice: Tuple[slice, slice], tile_slices: List[Tuple[slice, slice]]) -> List[Tuple[slice, slice]]:
    return [get_final_slice(roi_slice, tile_slice) for tile_slice in tile_slices]


def get_final_slice(roi_slice: Tuple[slice, slice], tile_slice: Tuple[slice, slice]) -> Tuple[slice, slice]:
    roi_rs, roi_cs = roi_slice
    rs, cs = tile_slice

    final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
    final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

    return final_rs, final_cs
