import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from zia import BASE_PATH
from zia.data_store import DataStore, ZarrGroups
from zia.path_utils import FileManager, ResultsDirectories, filter_factory

from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb, plot_pic


def calc_gradient(image: np.ndarray):
    # Calculate gradients using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=9)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=9)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Calculate the gradient direction (optional)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Normalize the gradient magnitude to [0, 255] (optional)
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the gradient magnitude to uint8 for visualization
    gradient_magnitude_uint8 = gradient_magnitude_normalized.astype(np.uint8)
    return gradient_magnitude_uint8


def hist(image: np.ndarray):
    # Flatten the image to a 1D array
    pixels = image.flatten()

    pixels = pixels[(pixels < 230) & (pixels > 20)]
    # Create a histogram
    plt.hist(pixels, bins=256, range=(0, 256), density=False, color='gray', alpha=0.7)

    # Set axis labels and title
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Image')


def watershed(image: np.ndarray) -> np.ndarray:
    # Calculate gradients using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    gradient_magnitude = np.uint8(gradient_magnitude)

    # Create markers for watershed algorithm (you can use any segmentation technique or manual annotations)
    # For simplicity, let's use thresholding here
    _, markers = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    markers = markers.astype(np.int32)

    # Apply the watershed algorithm
    markers = cv2.watershed(image, markers)

    # Colorize the segmented regions for visualization
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    segmented_image[markers == -1] = [0, 0, 255]  # Mark boundaries with red color
    return segmented_image


if __name__ == "__main__":
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(species="rat", subject="NOR-025", protein="cyp2e1")
    )

    # set the level for which the image should be created. Everything smaller than 4
    # gets memory intense
    level = PyramidalLevel.FIVE

    print(len(file_manager.get_images()))

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)

        for i, roi in enumerate(data_store.rois):
            mask_array = data_store.get_array(ZarrGroups.LIVER_MASK, i, level)

            roi_image = data_store.read_full_roi(i, level)
            image_array = np.array(roi_image)

            gs = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)

            gs[~mask_array[:]] = 255

            th, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            gs[gs > th] = 255
            gs[gs == 255] = 0
            plot_pic(gs)

            clahe = cv2.createCLAHE(2.0, tileGridSize=(8, 8))

            gs = clahe.apply(gs)
            gs = cv2.GaussianBlur(gs, (9, 9), 0)

            _, vessels = cv2.threshold(gs, 50, 255, cv2.THRESH_BINARY)

            vessels = vessels.reshape(vessels.shape[0], vessels.shape[1], 1).astype(np.uint32)
            plot_pic(vessels)


            gs_bgr = cv2.merge([gs, gs, gs])


            markers = cv2.watershed(gs_bgr, vessels)

            hist(gs)
            # gs = watershed(gs)

            plot_pic(gs)
