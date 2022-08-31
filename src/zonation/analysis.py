import numpy as np
from skimage.filters import gaussian, laplace


def histogram_normalization(image: np.ndarray) -> np.ndarray:
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
        quantiles = np.quantile(image_flat, q=[0.01, 0.99])

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
        image_hist[:, :, k] = (image_hist[:, :, k] - np.min(image_hist[:, :, k])) \
                              / (np.max(image_hist[:, :, k]) - np.min(image_hist[:, :, k]))

    # return renormalized data
    return image_hist


if __name__ == "__main__":
    from zonation import IMAGE_PATH
    from read_images import FluorescenceImage, Fluorophor
    from plots import plot_image_with_hist, plot_zonation, plot_overlay

    # plot image channels & histograms
    fimage = FluorescenceImage.from_file(IMAGE_PATH / "Test33.pickle")
    cyp2e1 = fimage.get_channel_data(Fluorophor.CY3)
    ecad = fimage.get_channel_data(Fluorophor.ALEXA_FLUOR_488)

    # Display raw data of channel
    image_raw = np.zeros(shape=(cyp2e1.shape[0], cyp2e1.shape[1], 2))
    image_raw[:, :, 0] = cyp2e1
    image_raw[:, :, 1] = ecad
    plot_image_with_hist(image_raw, cmap="gray", title="raw data")
    # FIXME: save

    # perform histogram normalization
    image_hist = histogram_normalization(image_raw)
    plot_image_with_hist(image_hist, cmap="gray", title="histogram processing")

    # calculate ratio
    plot_zonation(image_hist, cmap="tab20c", title="difference and ratio")

    # gauss normalization (depending on radius/structure)
    image_gauss = gaussian(image_hist, sigma=20, multichannel=True, channel_axis=None)
    # image_gauss = gaussian(image_gauss, sigma=10, channel_axis=1)
    plot_zonation(image_gauss, cmap="tab20c", title="gauss filtering for averaging")
    plot_overlay(image_hist, image_gauss, alpha=0.4)
