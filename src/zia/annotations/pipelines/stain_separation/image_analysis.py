import numpy as np
import zarr

from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import (
    calculate_optical_density,
    calculate_stain_matrix,
    deconvolve_image,
)
from zia.annotations.workflow_visualizations.util.image_plotting import (
    plot_pic,
    plot_rgb,
)
from zia.data_store import DataStore, ZarrGroups


class StainSeparator:
    """
    Stain separation for image in tiles.
    Stain matrix can be calculated for each roi on low level image
    However, the estimation of robust extrema has to be done globally

    """
    @classmethod
    def separate_stains(cls, data_store: DataStore) -> None:
        for i, roi in enumerate(data_store.rois):
            stain_matrix = cls.calculate_stain_matrix(
                data_store, i, roi, PyramidalLevel.SEVEN
            )
            cls.deconvolve_stains(data_store, i, stain_matrix)

    @classmethod
    def deconvolve_stains(
        cls, data_store: DataStore, roi_no: int, stain_matrix: np.ndarray
    ) -> None:

        dset: zarr.Array = data_store.get_array(ZarrGroups.LIVER_MASK, roi_no, PyramidalLevel.ZERO)

        # list of slice that slice the mask in square tiles of 2*13 pixels
        slices = get_tile_slices(shape=dset.shape)

        r, c = dset.shape

        # iterate over the tiles
        for rs, cs in slices:
            tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

            # read image tile from the open slide
            image_tile = data_store.read_region_from_roi(
                roi_no=roi_no,
                location=(cs.start, rs.start),
                level=PyramidalLevel.ZERO,
                size=(tile_shape[1], tile_shape[0]),
            )

            # convert PIL Image to np array
            image_tile = np.array(image_tile)

            # remove the alpha channel
            image_tile = image_tile[:, :, :-1]
            plot_rgb(image_tile, False)
            # calculate the optical density
            optical_density = calculate_optical_density(image_tile)

            # deconvolve the image
            _, _, stain_tile, _ = deconvolve_image(
                optical_density, image_tile.shape, stain_matrix
            )

            plot_pic(stain_tile)

    @classmethod
    def calculate_stain_matrix(
        cls, data_store: DataStore, roi_no: int, roi: Roi, level: PyramidalLevel
    ) -> np.ndarray:
        image = data_store.read_roi_from_slide(roi, level)
        mask = data_store.get_array(ZarrGroups.LIVER_MASK, roi_no, level)
        image_array = np.array(image)
        # remove the alpha channel
        image_array = image_array[:, :, :-1]
        image_array[~mask[:]] = (np.nan, np.nan, np.nan)
        plot_rgb(image_array, False)
        od = calculate_optical_density(image_array)
        stain_matrix = calculate_stain_matrix(od)

        return stain_matrix
