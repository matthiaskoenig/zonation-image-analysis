"""Module for running zonation analysis."""
import numpy as np
from skimage.filters import gaussian

from zia import CZI_IMAGES, CZI_PATH, RESULTS_PATH
from zia.czi_io import FluorescenceImage, Fluorophor
from zia.zonation.plots import plot_image_with_hist, plot_overlay, plot_zonation


def histogram_quantile_normalization(
    image: np.ndarray, qlower: float = 0.01, qupper: float = 0.99
) -> np.ndarray:
    """Remove outliers at high values & rescale histogram to 256 range.

    set lowest entries to zero entries to zero,
    remove max entries
    remove long tail of intensities
     input image is converted according to the conventions of img_as_float (Normalized
     first to values [-1.0 ; 1.0] or [0 ; 1.0] depending on dtype of input)

    """
    image_hist = image.copy()
    for k in range(image.shape[2]):
        # hist, bins = np.histogram(image[:, :, k], bins=256)
        image_flat = image[:, :, k].flatten()
        quantiles = np.quantile(image_flat, q=[qlower, qupper])

        for p in range(image.shape[0]):
            for q in range(image.shape[1]):
                value = image[p, q, k]
                if value < quantiles[0]:
                    image_hist[p, q, k] = quantiles[0]
                elif value > quantiles[1]:
                    # FIXME: this could remove important information for portal
                    # field detection
                    image_hist[p, q, k] = quantiles[1]

        # min-max normalization
        image_hist[:, :, k] = (image_hist[:, :, k] - np.min(image_hist[:, :, k])) / (
            np.max(image_hist[:, :, k]) - np.min(image_hist[:, :, k])
        )

    # return renormalized data
    return image_hist


def run_zonation_analysis(sid: str) -> None:
    """Run image analysis for given image sid."""
    print(f"--- {sid} ---")
    fimage = FluorescenceImage.from_file(CZI_PATH / f"{sid}.pickle")

    # TODO: check the assignments of proteins to the fluorophors
    cyp2e1 = fimage.get_channel_data(Fluorophor.ALEXA_FLUOR_488)
    ecad = fimage.get_channel_data(Fluorophor.CY3)
    dapi = fimage.get_channel_data(Fluorophor.DAPI)

    # Display raw data of channel
    if dapi is None:
        # some images do not have dapi information
        layers = 2
    else:
        layers = 3

    image_raw = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], layers))  # type: ignore
    image_raw[:, :, 0] = cyp2e1
    image_raw[:, :, 1] = ecad
    if dapi is not None:
        image_raw[:, :, 2] = dapi

    plot_image_with_hist(
        image_raw,
        cmap="gray",
        title=f"{sid}: raw data",
        path=RESULTS_PATH / f"raw_{sid}.png",
    )

    # perform histogram normalization
    image_hist = histogram_quantile_normalization(image_raw, qlower=0.01)
    plot_image_with_hist(
        image_hist,
        cmap="gray",
        title=f"{sid}: histogram processing",
        path=RESULTS_PATH / f"hist_{sid}.png",
    )

    # calculate and plot ratios
    cmap = "seismic"
    plot_zonation(
        image_hist,
        cmap=cmap,
        title=f"{sid}: difference and ratio",
        path=RESULTS_PATH / f"diffratio_{sid}.png",
    )

    # gauss normalization (depending on radius/structure)
    image_gauss = gaussian(image_hist, sigma=20, multichannel=True, channel_axis=None)
    # image_gauss = gaussian(image_gauss, sigma=10, channel_axis=1)
    plot_zonation(
        image_gauss,
        cmap=cmap,
        title=f"{sid}: gauss filtering for averaging",
        path=RESULTS_PATH / f"gauss_{sid}.png",
    )

    # calculate zonation measures & plot the overlays
    plot_overlay(
        image_hist,
        image_gauss,
        zonation=image_hist[:, :, 0] - image_hist[:, :, 1],
        zonation_gauss=image_gauss[:, :, 0] - image_gauss[:, :, 1],
        cmap=cmap,
        alpha=0.4,
        title=f"{sid}: Zonation Difference (CYP2E1 - E-Cadherin)",
        path=RESULTS_PATH / f"overlay_difference_{sid}.png",
    )
    plot_overlay(
        image_hist,
        image_gauss,
        zonation=image_hist[:, :, 0] / image_hist[:, :, 1],
        zonation_gauss=image_gauss[:, :, 0] / image_gauss[:, :, 1],
        cmap=cmap,
        alpha=0.4,
        title=f"{sid}: Zonation Ratio (CYP2E1 / E-Cadherin)",
        path=RESULTS_PATH / f"overlay_ratio_{sid}.png",
    )

    # TODO: gauss filter based on geometry (resolution), i.e. different meaning depending on resolution
    # TODO: area quantification via hist (20 areas), i.e. a distribution
    # TODO: store results in image for visualization

    # Display raw data of channel
    image_out = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], 5))  # type: ignore
    image_out[:, :, 0] = image_hist[:, :, 0]
    image_out[:, :, 1] = image_hist[:, :, 1]
    if dapi is not None:
        image_out[:, :, 2] = image_hist[:, :, 2]
    # zonation difference
    image_out[:, :, 3] = image_hist[:, :, 0] - image_hist[:, :, 1]
    image_out[:, :, 4] = image_hist[:, :, 0] - image_hist[:, :, 1]

    # TODO: save image as tif with zonation channels


if __name__ == "__main__":

    for p in CZI_IMAGES:
        sid = p.stem
        run_zonation_analysis(sid)

    # run_analysis("Test33")
