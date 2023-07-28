"""Using DAB channel from stain separation to create a mesh
which can be manipulated with meshio.

In geometry, a triangulation is a subdivision of a planar object into triangles,
and by extension the subdivision of a higher-dimension geometric object into
simplices. Triangulations of a three-dimensional volume would involve
subdividing it into tetrahedra packed together.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import calculate_stain_matrix, \
    deconvolve_image
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic, \
    plot_rgb
from zia.config import read_config
from zia.data_store import DataStore, ZarrGroups
from zia.path_utils import FileManager, ImageInfo
from zia.console import console


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


def plot_array(data: np.ndarray):
    """Plot single array."""
    f, ax = plt.subplots(1, 1, dpi=300)
    ax.imshow(data, cmap='gray')
    ax.axis("off")
    f.tight_layout()
    plt.show()


def reconstruct(idx, pxi, shape) -> np.ndarray:
    """Reconstruction of image."""
    new_image = np.ones(shape=shape).astype(np.uint8) * 255
    new_image[idx[:, 0], idx[:, 1], :] = pxi
    return new_image


def separate_channels(
    image_info: ImageInfo, level: PyramidalLevel=PyramidalLevel.THREE
) -> dict[str, np.ndarray]:
    """Get the DAB channel information for the mesh."""
    roi_no = 1

    data_store = DataStore(image_info)
    mask = data_store.get_array(ZarrGroups.LIVER_MASK, roi_no=roi_no, level=level)

    # read the full ROI
    region = data_store.read_full_roi(roi_no, level)

    # create numpy array of the image
    image_array = np.array(region)

    h, w, c = image_array.shape

    # convert RGBA to RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # threshold the image using otsu threshold
    threshold, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # selecting interesting pixels by mask and OTSU threshold
    pxi = image_array[mask[:] & (gs < threshold)]

    # index of interesting pixels
    idx = np.argwhere(mask[:] & (gs < threshold))

    # calculate stain matrix
    stain_matrix = calculate_stain_matrix(pxi)

    # calculate images
    Io, he, dab = deconvolve_image(pxi, stain_matrix)

    dab_image: np.ndarray = reconstruct(idx, dab, shape=(h, w, 1))
    he_image: np.ndarray = reconstruct(idx, he, shape=(h, w, 1))
    # io_image: np.ndarray = reconstruct(idx, Io, shape=(h, w, 3))

    return {
        "he": he_image,
        "dab": dab_image,
    }

def array_info(data: np.ndarray):
    """Print array information."""
    console.print(f"{data.dtype}, {data.shape}")

def channels_info(channels):
    """Log channels information."""
    for channel_key, channel_data in channels.items():
        console.print(f"{channel_key=}")
        array_info(channel_data)


if __name__ == "__main__":
    from zia import BASE_PATH

    dab_path = 'dab_test_L2.npy'
    level = PyramidalLevel.TWO

    if False:
        image_name = (
            "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006"
        )

        # TODO: get DAB channel in highest resolution
        file_manager = FileManager(
            configuration=read_config(BASE_PATH / "configuration.ini"),
            filter=None
        )

        image_infos: list[ImageInfo] = file_manager.get_images()

        image_infos = list(filter(lambda x: x.path.stem == image_name, image_infos))
        image_info: ImageInfo = image_infos[0]

        channels: dict[str, np.ndarray] = separate_channels(
            image_info=image_info,
            level=level,
        )
        channels_info(channels)

        # serialization
        np.save(dab_path, channels["dab"])


    console.rule(title="serialization", style="white")
    dab_data = np.load(dab_path)
    array_info(dab_data)
    plot_array(dab_data)

    # TODO: median filter

    # TODO: generate mesh from image
    # meshio:

    # pyvista: Delauny triangulation
    """
    import numpy as np
    import pyvista as pv

    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    cloud = pv.PolyData(points)
    cloud.plot()

    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()
    shell.plot()
    """



