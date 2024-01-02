from typing import Callable, Tuple

import numpy as np
import cv2
from zia.pipeline.common.resolution_levels import PyramidalLevel




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
        img[(img < lower)] = lower
        img[(img > upper)] = upper
        return img

    def _filter_img(self, img) -> np.ndarray:
        # print(img.shape)
        img = self._convolute_median(img, ksize=7)
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
        return PyramidalLevel.get_by_numeric_level(round(new_level))

    def prepare_image(self) -> Tuple[int, np.ndarray]:
        self._apply_filter(self._invert_image)
        self._set_missing_to_zero()
        self._apply_filter(self._filter_img)
        return self._get_level_diff(), self.image
