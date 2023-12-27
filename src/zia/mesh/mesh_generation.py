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

from zia.pipeline.annotation import PyramidalLevel
from zia.pipeline.pipeline_components.algorithm.stain_separation.macenko import calculate_stain_matrix, \
    deconvolve_image
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
    nx, ny, _ = data.shape
    ratio = int(np.floor(nx/ny))
    # create image which fits to the ratio
    f, ax = plt.subplots(1, 1, dpi=300, figsize=(10, 10*ratio))
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
    roi_no = 0

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

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None
    )

    image_infos: list[ImageInfo] = file_manager.get_images()
    console.print(image_infos)

    # Overview figures for single stains
    # title = "MNT-025"
    title = "UKJ-19-010_Human"
    images = {
        # # "HE": "MNT-025_Bl6J_J-20-0160_HE_Run 05_LLL, RML, RSL, ICL_MAA_0003",
        # "GS": "MNT-025_Bl6J_J-20-0160_GS 1 1000_Run 06_LLL, RML, RSL, ICL_MAA_0001",
        # "CYP2E1": "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006",
        # "CYP1A2": "MNT-025_Bl6J_J-20-0160_CYP1A2-1 500_Run 08_LLL, RML, RSL, ICL_MAA_0002",
        # "CYP3A4": "MNT-025_Bl6J_J-20-0160_CYP3A4 1 2000_Run 10_LLL, RML, RSL, ICL_MAA_0005",
        # "CYP2D6": "MNT-025_Mouse_Bl6J_J-20-0160_CYP2D6- 1 3000_Run 14_ML, RML, RSL, ICL_MAA_004",

        # "HE": "UKJ-19-010_Human _J-19-0154_HE_Run 07__MAA_003",
        "GS": "UKJ-19-010_Human _J-19-0154_GS 1 1000_Run 09_Liver_MAA_001",
        "CYP2E1": "UKJ-19-010_Human _J-19-0154_CYP2E1-1 800_Run 15__MAA_006",
        "CYP1A2": "UKJ-19-010_Human _J-19-0154_CYP1A2-1 2000_Run 12__MAA_002",
        "CYP3A4": "UKJ-19-010_Human _J-19-0154_CYP3A4 1 1500_Run 14__MAA_005",
        "CYP2D6": "UKJ-19-010_Human _J-19-0154_CYP2D6- 1 200_Run 19__MAA_007",
    }

    all_channels = {}
    for staining, image_name in images.items():
        image_infos: list[ImageInfo] = file_manager.get_images()
        image_infos = list(filter(lambda x: x.path.stem == image_name, image_infos))
        image_info: ImageInfo = image_infos[0]

        channels: dict[str, np.ndarray] = separate_channels(
            image_info=image_info,
            level=PyramidalLevel.FIVE,
        )
        channels_info(channels)
        all_channels[staining] = channels

    """Plot data"""
    # create image which fits to the ratio
    n_stainings = len(all_channels)
    f, axes = plt.subplots(1, n_stainings, dpi=300, figsize=(5*n_stainings, 5))
    f.suptitle(title)
    for k, staining in enumerate(all_channels):
        channels = all_channels[staining]
        data = channels["dab"]
        axes[k].set_title(staining)
        axes[k].imshow(data, cmap='gray')
        axes[k].axis("off")
    f.tight_layout()
    plt.show()
    f.savefig(f"{title}_stain_separation.png")
    exit()


    if True:
        image_name = (
            "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006"
        )

        image_infos = list(filter(lambda x: x.path.stem == image_name, image_infos))
        image_info: ImageInfo = image_infos[0]

        for dab_path, level in [
            ('dab_test_L7.npy', PyramidalLevel.SEVEN),
            ('dab_test_L6.npy', PyramidalLevel.SIX),
            ('dab_test_L5.npy', PyramidalLevel.FIVE),
            ('dab_test_L4.npy', PyramidalLevel.FOUR),
            ('dab_test_L3.npy', PyramidalLevel.THREE),
            ('dab_test_L2.npy', PyramidalLevel.ONE),
            ('dab_test_L1.npy', PyramidalLevel.ZERO),
            ('dab_test_L0.npy', PyramidalLevel.ZERO),
        ]:
            channels: dict[str, np.ndarray] = separate_channels(
                image_info=image_info,
                level=level,
            )
            channels_info(channels)
            # serialization
            np.save(dab_path, channels["dab"])

    dab_path = 'dab_test_L7.npy'
    console.rule(title="serialization", style="white")
    dab_data = np.load(dab_path)
    array_info(dab_data)
    plot_array(dab_data)

    # TODO: median filter
    # apply filtering for smoother output

    # TODO: generate mesh from image

    import numpy as np
    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    # convert the data to points
    n_x = dab_data.shape[0]
    n_y = dab_data.shape[1]
    n_points = n_x * n_y
    points = np.zeros(shape=(n_points, 3))
    kp = 0
    for kx in range(n_x):
        for ky in range(n_y):
            points[kp, 0] = kx
            points[kp, 1] = ky
            points[kp, 2] = dab_data[kx, ky, 0]

    # scipy triangulation
    console.rule(title="triangulation", style="white")
    from scipy.spatial import Delaunay
    tri = Delaunay(points)

    import matplotlib.pyplot as plt
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    if False:
        # pyvista triangulation
        import pyvista as pv
        console.print("cloud")
        cloud = pv.PolyData(points)
        cloud.plot()

        console.print("volume")
        volume = cloud.delaunay_3d(alpha=2.)
        # shell = volume.extract_geometry()

    # shell.plot()
