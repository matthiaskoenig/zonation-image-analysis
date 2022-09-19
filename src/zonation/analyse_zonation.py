import numpy as np
from skimage.filters import gaussian
from zonation import IMAGE_PATH, RESULTS_PATH, CZI_IMAGES
from read_images import FluorescenceImage, Fluorophor
from plots import plot_image_with_hist, plot_zonation, plot_overlay
from zonation.analysis import histogram_quantile_normalization


def run_analysis(sid: str, show_plot: bool = True):
    """Run image analysis for given image sid."""
    print(f"--- {sid} ---")
    fimage = FluorescenceImage.from_file(IMAGE_PATH / f"{sid}.pickle")

    # TODO: check the assignments of proteins to the fluorophors
    cyp2e1 = fimage.get_channel_data(Fluorophor.ALEXA_FLUOR_488)
    ecad = fimage.get_channel_data(Fluorophor.CY3)
    dapi = fimage.get_channel_data(Fluorophor.DAPI)

    # Display raw data of channel
    if dapi is not None:
        image_raw = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], 3))
        image_raw[:, :, 2] = dapi
    else:
        image_raw = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], 2))
    image_raw[:, :, 0] = cyp2e1
    image_raw[:, :, 1] = ecad

    plot_image_with_hist(image_raw, cmap="gray", title=f"{sid}: raw data",
                         path=RESULTS_PATH / f"raw_{sid}.png")

    # perform histogram normalization
    image_hist = histogram_quantile_normalization(image_raw, qlower=0.01)
    plot_image_with_hist(image_hist, cmap="gray", title=f"{sid}: histogram processing", path=RESULTS_PATH / f"hist_{sid}.png")

    # calculate and plot ratios
    cmap = "seismic"
    plot_zonation(image_hist, cmap=cmap, title=f"{sid}: difference and ratio", path=RESULTS_PATH / f"diffratio_{sid}.png")

    # gauss normalization (depending on radius/structure)
    image_gauss = gaussian(image_hist, sigma=20, multichannel=True, channel_axis=None)
    # image_gauss = gaussian(image_gauss, sigma=10, channel_axis=1)
    plot_zonation(image_gauss, cmap=cmap, title=f"{sid}: gauss filtering for averaging", path=RESULTS_PATH / f"gauss_{sid}.png")

    # calculate zonation measures & plot the overlays
    plot_overlay(
        image_hist, image_gauss,
        zonation=image_hist[:, :, 0] - image_hist[:, :, 1],
        zonation_gauss=image_gauss[:, :, 0] - image_gauss[:, :, 1],
        cmap=cmap, alpha=0.4,
        title=f"{sid}: Zonation Difference (CYP2E1 - E-Cadherin)",
        path=RESULTS_PATH / f"overlay_difference_{sid}.png"
    )
    plot_overlay(
        image_hist, image_gauss,
        zonation=image_hist[:, :, 0] / image_hist[:, :, 1],
        zonation_gauss=image_gauss[:, :, 0] / image_gauss[:, :, 1],
        cmap=cmap, alpha=0.4,
        title=f"{sid}: Zonation Ratio (CYP2E1 / E-Cadherin)",
        path=RESULTS_PATH / f"overlay_ratio_{sid}.png"
    )

    # TODO: gauss filter based on geometry (resolution), i.e. different meaning depending on resolution
    # TODO: area quantification via hist (20 areas), i.e. a distribution
    # TODO: store results in image for visualization

    # Display raw data of channel
    image_out = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], 5))
    image_out[:, :,0] = image_hist[:, :, 0]
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
       run_analysis(sid)

    # run_analysis("Test33")



