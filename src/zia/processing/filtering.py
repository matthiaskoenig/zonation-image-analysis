import matplotlib.pyplot as plt
import numpy as np
import zarr

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.data_store import ZarrGroups
from imagecodecs.numcodecs import Jpegxl, Jpeg2k
import numcodecs
from sklearn.cluster import KMeans

numcodecs.register_codec(Jpeg2k)
import cv2

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.FOUR


def convolute_meadian(img: np.ndarray, ksize=3) -> np.ndarray:
    median_blurr = cv2.medianBlur(img, ksize)
    return median_blurr[::2, ::2]


def adaptive_hist_norm(img: np.ndarray, clip_limit=2.0, ksize=(16, 16)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=ksize)
    img[img < 10] = 0  # removing the background that got changed by the adapative norm

    return clahe.apply(img)


def cut_off_percentile(img: np.ndarray, p: float):
    lower = np.percentile(img[img > 0], p)
    upper = np.percentile(img[img > 0], (1 - p) * 100)
    img[(img < lower) & (img > upper)] = 0
    return img


def filter_img(img) -> np.ndarray:
    print(img.shape)
    img = convolute_meadian(img, ksize=7)
    img = cut_off_percentile(img, 0.05)
    img = adaptive_hist_norm(img, ksize=(8, 8))
    img = convolute_meadian(img, ksize=3)
    img = (img / np.max(img)) * 255
    return img.astype(np.uint8)


def invert_image(img: np.ndarray) -> np.ndarray:
    return 255 - img


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        print(i)
        if i in ["HE", "CYP2D6"]:
            continue
        img = np.array(a.get(f"{level}"))
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        arrays[i] = img  # cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, mask=mask)

    conv = {i: invert_image(a) for i, a in arrays.items()}

    merged = np.stack(list(conv.values()), axis=-1)

    # remove non overlapping pixels
    mask = np.any(merged[:, :, :] == 0, axis=-1)
    merged[mask, :] = 0

    print(merged.shape)
    # apply filters

    filtered = np.stack([filter_img(merged[:, :, i]) for i in range(merged.shape[2])], axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=600)
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 6), dpi=600)

    for i, (key, ax, ax1) in enumerate(zip(arrays.keys(), axes.flatten(), axes1.flatten())):
        arr = filtered[:, :, i]
        hist = arr[arr > 0]

        p = 0.05

        """ lower = np.percentile(hist, p)
        upper = np.percentile(hist, (1 - p) * 100)
        print(lower, upper)
        hist = arr[(arr > lower) & (arr < upper)]"""

        # arr[(arr < lower) & (arr > upper)] = 0
        ax.hist(hist, bins="sqrt")
        ax1.imshow(arr, cmap="binary_r")
        ax: plt.Axes
        ax.set_title(key)
        ax1.set_title(key)

    plt.show()
