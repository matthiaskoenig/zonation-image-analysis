from typing import List

import numpy as np
import dask.array as da
import napari


def view_ndpi_data(data: List[da.Array]) -> None:
    """View NDPI image in napari.

    This is starting napari and blocking.
    """
    viewer = napari.Viewer()
    viewer.add_image(data)
    napari.run()


def view_czi_data(data: np.ndarray) -> None:
    """View CZI image in napari.

    This is starting napari and blocking.
    """
    viewer = napari.Viewer()
    viewer.add_image(data)
    napari.run()
