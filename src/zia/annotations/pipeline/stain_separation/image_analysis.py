import cv2
import numpy as np
import zarr
from PIL.Image import fromarray

from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.open_slide_image.data_store import DataStore, ZarrGroups
from zia.annotations.pipeline.stain_separation.macenko import calculate_stain_matrix, calculate_optical_density, deconvolve_image
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb, plot_pic


class StainSeparator:

    @classmethod
    def separate_stains(cls, data_store: DataStore) -> None:
        for i, roi in enumerate(data_store.rois):
            stain_matrix = cls.calculate_stain_matrix(data_store, i, roi, PyramidalLevel.SEVEN)
            cls.deconvolve_stains(data_store, i, stain_matrix)

    @classmethod
    def deconvolve_stains(cls, data_store: DataStore, roi_no: int, stain_matrix: np.ndarray) -> None:
        # list of slice that slice the mask in square tiles of 2*13 pixels
        dset: zarr.Array = data_store.data.get(f"{ZarrGroups.LIVER_MASK}/{roi_no}/{0}")
        slices = get_tile_slices(shape=dset.shape)

        r, c = dset.shape

        # iterate over the tiles
        for rs, cs in slices:
            tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

            # read image tile from the open slide
            image_tile = data_store.read_region_from_roi(roi_no=roi_no,
                                                         location=(cs.start, rs.start),
                                                         level=PyramidalLevel.ZERO,
                                                         size=(tile_shape[1], tile_shape[0]))

            # convert PIL Image to np array
            image_tile = np.array(image_tile)

            # remove the alpha channel
            image_tile = image_tile[:, :, :-1]
            plot_rgb(image_tile, False)
            # calculate the optical density
            optical_density = calculate_optical_density(image_tile)

            # deconvolve the image
            _, _, stain_tile, _ = deconvolve_image(optical_density, image_tile.shape, stain_matrix)

            plot_pic(stain_tile)

    @classmethod
    def calculate_stain_matrix(cls, data_store: DataStore, roi_no: int, roi: Roi, level: PyramidalLevel) -> np.ndarray:
        image = data_store.read_roi_from_slide(roi, level)
        mask = cls.down_scale_mask_easy(data_store, roi_no, level)
        image_array = np.array(image)
        # remove the alpha channel
        image_array = image_array[:, :, :-1]
        image_array[~mask] = (255, 255, 255)
        plot_rgb(image_array, False)
        od = calculate_optical_density(image_array)
        stain_matrix = calculate_stain_matrix(od)

        return stain_matrix

    @classmethod
    def down_scale_mask(cls, data_store: DataStore, roi_no: int, level: PyramidalLevel) -> np.ndarray:
        dset: zarr.Array = data_store.data.get(f"{ZarrGroups.LIVER_MASK}/{roi_no}/{0}")

        # new height and width of the downsized mask
        new_shape = tuple([int(x / 2 ** level) for x in dset.shape])

        down_sized_mask = np.empty(shape=new_shape)
        print(f"downsized_mask {down_sized_mask.shape}")

        # list of slice that slice the mask in square tiles of 2*13 pixels
        slices = get_tile_slices(shape=dset.shape)

        # iterate over the tiles
        for rs, cs in slices:
            # creates a binary image from the slice of the boolean zarr array
            mask_tile = (dset[rs, cs] * 255).astype(np.uint8)

            # calculate slices for downsized array
            new_rs, new_cs = [slice(int(s.start / 2 ** level), int(s.stop / 2 ** level)) for s in (rs, cs)]

            # swap with and height for open cv
            new_tile_shape = (new_cs.stop - new_cs.start, new_rs.stop - new_rs.start)

            # resize the tile
            print(f"new tile shape {new_tile_shape}")
            down_sized_tile = cv2.resize(mask_tile, new_tile_shape)
            print(f"downsized_tile {down_sized_tile.shape}")

            # set tile on resulting mask
            down_sized_mask[new_rs, new_cs] = down_sized_tile

        return down_sized_mask.astype(bool)

    @classmethod
    def down_scale_mask_easy(cls, data_store: DataStore, roi_no: int, level: PyramidalLevel):
        dset: zarr.Array = data_store.data.get(f"{ZarrGroups.LIVER_MASK}/{roi_no}/{0}")
        factor = 2 ** level
        return dset[::factor, ::factor]
