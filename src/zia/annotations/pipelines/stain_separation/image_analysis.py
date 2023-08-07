import queue
import threading
from tempfile import TemporaryFile
import time
from tempfile import TemporaryFile
from typing import Tuple, List

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


class StainSeparator:
    """
    Stain separation for image in tiles.
    Stain matrix can be calculated for each roi on low level image
    However, the estimation of robust extrema has to be done globally

    """

    @classmethod
    def separate_stains(cls, data_store: DataStore, p=0.01, level=PyramidalLevel.ZERO) -> None:
        arrays = read_ndpi(data_store.image_info.path)
        full_image_array: zarr.Array = arrays[level]

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

            slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=2 ** 13)

            samples = []

            # iterate over the tiles
            t_s = time.time()
            for tile_slice in slices:
                samples.append(
                    cls.calculate_samples_for_otsu(tile_slice, (roi_rs, roi_cs),
                                                   full_image_array, p))

            dask.compute(samples)

            samples = np.hstack(samples).astype(np.uint8)

            th, _ = cv2.threshold(samples, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            t_e = time.time()

            logger.info(f"Calculate Otsu threshold {th} in {(t_e - t_s) / 60} min.")

            t_s = time.time()

            # create pyramidal group to persist mask
            # pyramid_dict = data_store.create_pyramid_group(ZarrGroups.LIVER_MASK, roi_no, shape, bool)

            pxi_samples = []
            temp_files = []

            for i, (rs, cs) in enumerate(slices):

                final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
                final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

                tile_mask = mask[rs, cs]
                # read image tile from the open slide

                image_tile = full_image_array[final_rs, final_cs]

                # pixels in mask and otsu filtered

                idx = tile_mask & (np.dot(image_tile, np.array([0.587,
                                                                0.114,
                                                                0.299])) < th)

                temp_file = TemporaryFile()
                np.save(temp_file, idx, allow_pickle=True)
                temp_files.append(temp_file)

                px_in_mask = image_tile[idx]

                if len(px_in_mask) == 0:
                    continue

                choice = np.random.choice(a=[True, False], size=len(px_in_mask),
                                          p=[p, 1 - p])

                px_sample = px_in_mask[choice]

                pxi_samples.append(px_sample)

            pxi_samples = np.vstack(pxi_samples)

            stain_matrix = calculate_stain_matrix(pxi_samples)

            maxC = find_max_c(pxi_samples, stain_matrix)

            print(stain_matrix, maxC)

            t_e = time.time()

            print(
                f"Calculated stain matrix and max concentrations in {(t_e - t_s) / 60} min.")

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

    @classmethod
    def calculate_samples_for_otsu(cls, tile_slices: Tuple[slice, slice],
                                   roi_slices: Tuple[slice, slice],
                                   image_array: zarr.Array,
                                   p):
        roi_rs, roi_cs = roi_slices
        rs, cs = tile_slices

        final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
        final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

        tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

        t_ls = time.time()
        image_tile = image_array[final_rs, final_cs]
        t_le = time.time()
        # print(f"Loaded tile in {(t_le - t_ls)} s.")
        choice = np.random.choice(a=[True, False],
                                  size=tile_shape,
                                  p=[p, 1 - p])

        sample = image_tile[choice]
        sample_gs = sample.dot(np.array([0.587, 0.114, 0.299]))
        return sample_gs


