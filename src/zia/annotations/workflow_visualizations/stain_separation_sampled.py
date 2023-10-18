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
from zia.path_utils import FileManager


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


def plot_all(original, reconstructed, he, dab):
    fig, axes = plt.subplots(2, 2, dpi=300)
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("original")
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title("reconstructed")
    axes[1, 0].imshow(he, cmap="binary_r")
    axes[1, 0].set_title("HE")
    axes[1, 1].imshow(dab, cmap="binary_r")
    axes[1, 1].set_title("DAB")

    for ax in axes.flatten():
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def reconstruct(idx, pxi, shape) -> np.ndarray:
    new_image = np.ones(shape=shape).astype(np.uint8) * 255
    new_image[idx[:, 0], idx[:, 1], :] = pxi
    return new_image


if __name__ == "__main__":
    from zia import BASE_PATH

    image_name = (
        "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006"
    )

    roi_no = 2
    # level = PyramidalLevel.THREE

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None
    )

    image_infos = file_manager.get_images()

    image_infos = list(filter(lambda x: x.path.stem == image_name, image_infos))
    image_info = image_infos[0]

    data_store = DataStore(image_info)

    level = PyramidalLevel.THREE

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

    otsu_pxi_oi = gs.reshape(-1, 1)  # gs[mask[:]]
    print(otsu_pxi_oi.shape)

    # draw sample from the image and calculate threshold
    for p in [0.001, 0.01, 0.1, 1]:
        print(80 * "-")
        print(p)

        choice = np.random.choice(a=[True, False], size=len(otsu_pxi_oi), p=[p, 1 - p])

        otsu_sample = otsu_pxi_oi[choice].reshape(-1, 1)

        threshold, _ = cv2.threshold(otsu_sample, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        px_oi = image_array[mask[:] & (gs < threshold)]
        idx = np.argwhere(mask[:] & (gs < threshold))

        choice = np.random.choice(a=[True, False], size=len(px_oi), p=[p, 1 - p])

        px_sample = px_oi[choice]

        stain_matrix = calculate_stain_matrix(px_sample)

        print(stain_matrix)
        [Io_sample, he_sample, dab_sample], maxC = deconvolve_image(px_sample,
                                                                    stain_matrix)

        [Io_rest, he_rest, dab_rest], _ = deconvolve_image(px_oi[~choice], stain_matrix,
                                                           maxC=maxC)

        pxi_template = np.empty(shape=(len(px_oi), 1))

        pxi_template[choice] = dab_sample
        pxi_template[~choice] = dab_rest

        final_dab = reconstruct(idx=idx, pxi=pxi_template, shape=(h, w, 1))

        plot_pic(final_dab, f"Dab: p={p}")

"""
# threshold the image using otsu threshold
    threshold, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(threshold)
    # selecting interesting pixels by mask and OTSU threshold
    pxi = image_array[mask[:] & (gs < threshold)]

    # index of interesting pixels
    idx = np.argwhere(mask[:] & (gs < threshold))

    # calculate stain matrix
    stain_matrix = calculate_stain_matrix(pxi)

    print(stain_matrix)

    # calculate images
    Io, he, dab = deconvolve_image(pxi, stain_matrix)

    dab_image = reconstruct(idx, dab, shape=(h, w, 1))
    results_path = file_manager.results_path / "algo"
    results_path.mkdir(exist_ok=True)
    file_name = results_path / f'dab_{level}.png'
    print(file_name.__str__())
    cv2.imwrite(file_name.__str__(), dab_image)
    he_image = reconstruct(idx, he, shape=(h, w, 1))
    io_image = reconstruct(idx, Io, shape=(h, w, 3))

    plot_all(image_array, io_image, he_image, dab_image)"""
