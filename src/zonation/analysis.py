import numpy as np


def histogram_quantile_normalization(image: np.ndarray, qlower: float=0.01, qupper: float=0.99) -> np.ndarray:
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
        image_hist[:, :, k] = (image_hist[:, :, k] - np.min(image_hist[:, :, k])) \
                              / (np.max(image_hist[:, :, k]) - np.min(image_hist[:, :, k]))

    # return renormalized data
    return image_hist