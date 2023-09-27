from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpeg2k

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.processing.load_image_stack import load_image_stack_from_zarr

numcodecs.register_codec(Jpeg2k)
import cv2


class Filter:
    def __init__(self, image: np.ndarray, level: PyramidalLevel):
        self.incoming_shape = image.shape
        self.image = image
        self.level = level

    def _convolute_median(self, img: np.ndarray, ksize=3) -> np.ndarray:
        median_blurr = cv2.medianBlur(img, ksize)
        return median_blurr[::2, ::2]

    def _adaptive_hist_norm(self, img: np.ndarray, clip_limit=2.0, ksize=(16, 16)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=ksize)
        norm = clahe.apply(img)
        norm[norm < 10] = 0  # removing the background that got changed by the adapative norm

        return norm

    def _cut_off_percentile(self, img: np.ndarray, p: float):
        lower = np.percentile(img[img > 0], p)
        upper = np.percentile(img[img > 0], (1 - p) * 100)
        img[(img < lower) & (img > upper)] = 0
        return img

    def _filter_img(self, img) -> np.ndarray:
        # print(img.shape)
        img = self._convolute_median(img, ksize=7)
        img = self._cut_off_percentile(img, 0.05)
        img = self._adaptive_hist_norm(img, ksize=(8, 8))
        img = self._convolute_median(img, ksize=3)
        img = (img / np.max(img)) * 255
        return img.astype(np.uint8)

    @classmethod
    def _invert_image(cls, img: np.ndarray) -> np.ndarray:
        return 255 - img

    def _set_missing_to_zero(self):
        mask = np.any(self.image[:, :, :] == 0, axis=-1)
        self.image[mask, :] = 0

    def _apply_filter(self, fun: Callable[[np.ndarray], np.ndarray]) -> None:
        self.image = np.stack([fun(self.image[:, :, i]) for i in range(self.image.shape[2])], axis=-1)

    def _get_level_diff(self):
        diff = np.log2(self.incoming_shape[0] / self.image.shape[0])
        new_level = self.level + diff
        return PyramidalLevel.get_by_numeric_level(int(new_level))

    def prepare_image(self) -> Tuple[int, np.ndarray]:
        self._apply_filter(self._invert_image)
        self._set_missing_to_zero()
        self._apply_filter(self._filter_img)
        return self._get_level_diff(), self.image


if __name__ == "__main__":
    #subject = "SSES2021 9"
    roi = "0"
    subject = "UKJ-19-041_Human"
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    loaded_level, image = load_image_stack_from_zarr(group, max_size=2.5e7)
    print(loaded_level)
    filter = Filter(image, loaded_level)

    final_level, filtered_image = filter.prepare_image()
    print(final_level)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=600)
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 6), dpi=600)

    for i, (key, ax, ax1) in enumerate(zip(group.keys(), axes.flatten(), axes1.flatten())):
        arr = filtered_image[:, :, i]
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
