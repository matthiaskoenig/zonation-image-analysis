from typing import Tuple

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


def separate_stains(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # create a grayscale representation to find interesting pixels with otsu
    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    otsu_sample = gs.reshape(-1, 1)  # gs[mask[:]]

    threshold, _ = cv2.threshold(otsu_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    idx = gs < threshold
    px_oi = image_array[idx]

    stain_matrix = calculate_stain_matrix(px_oi)

    concentrations = deconvolve_image_and_normalize(image_array.reshape(-1, 3), stain_matrix, maxCRef=None)

    h, e = create_single_channel_pixels(concentrations)

    return h.reshape(gs.shape), e.reshape(gs.shape)


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


import argparse
import numpy as np
from PIL import Image


def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    print(np.min(C2), np.max(C2))

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    print(np.min(H), np.max(H))

    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))

    print(np.min(E), np.max(E))
    print(np.min(H), np.max(H))

    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.png')
        Image.fromarray(H).save(saveFile + '_H.png')
        Image.fromarray(E).save(saveFile + '_E.png')

    return Inorm, H, E
