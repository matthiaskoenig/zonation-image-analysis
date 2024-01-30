import cv2
import numpy as np

from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.pipeline_components.algorithm.stain_separation.macenko import calculate_stain_matrix, \
    deconvolve_image, find_max_c, create_single_channel_pixels, reconstruct_pixels, deconvolve_image_and_normalize


def normalize_stain1(image_array: np.ndarray, HERef: np.ndarray) -> np.ndarray:

    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_sample = gs.reshape(-1, 1)  # gs[mask[:]]

    threshold, _ = cv2.threshold(otsu_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(px_oi, stain_matrix)

    normalized = reconstruct_pixels(concentrations=concentrations, refrence_matrix=HERef)

    template = image_array.copy()

    template[idx] = normalized
    return template


def normalize_stain(image_array: np.ndarray, HERef: np.ndarray, maxCRef: np.ndarray) -> np.ndarray:
    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_sample = gs.reshape(-1, 1)  # gs[mask[:]]

    threshold, _ = cv2.threshold(otsu_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(image_array.reshape(-1, 3), stain_matrix, maxCRef)

    normalized = reconstruct_pixels(concentrations=concentrations, refrence_matrix=HERef)

    return normalized.reshape(image_array.shape)

