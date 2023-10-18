import cv2
import numpy as np

from zia.annotations.pipelines.stain_separation.macenko import calculate_stain_matrix, \
    deconvolve_image, find_max_c, create_single_channel_pixels


def separate_raw_image(image_array: np.ndarray, sampling_rate=0.01):
    # convert RGBA to RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_pxi_oi = gs.reshape(-1, 1)  # gs[mask[:]]

    p = sampling_rate
    # draw sample from the image and calculate threshold

    choice = np.random.choice(a=[True, False], size=len(otsu_pxi_oi), p=[p, 1 - p])

    otsu_sample = otsu_pxi_oi[choice].reshape(-1, 1)

    threshold, _ = cv2.threshold(otsu_sample, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    choice = np.random.choice(a=[True, False], size=len(px_oi), p=[p, 1 - p])

    px_sample = px_oi[choice]
    px_rest = px_oi[~choice]

    stain_matrix = calculate_stain_matrix(px_sample)

    max_c = find_max_c(px_sample, stain_matrix)

    # reconstruction
    final = []
    for k in range(2):
        c_sample = deconvolve_image(px_sample, stain_matrix, max_c)

        stain_sample = create_single_channel_pixels(c_sample[k, :])

        c_rest = deconvolve_image(px_rest, stain_matrix, maxC=max_c)

        stain_rest = create_single_channel_pixels(c_rest[k, :])

        pxi_oi_template = np.empty(shape=px_oi.shape[0])

        pxi_oi_template[choice] = stain_sample
        pxi_oi_template[~choice] = stain_rest

        image_template = np.ones(shape=image_array.shape[:2]).astype(np.uint8) * 255
        image_template[idx] = pxi_oi_template
        final.append(image_template)

    return final
