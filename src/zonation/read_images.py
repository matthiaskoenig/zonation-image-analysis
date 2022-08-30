"""Reading images.

Using bioformats with python-javabridge

https://pypi.org/project/python-javabridge/
https://pythonhosted.org/javabridge/installation.html

git clone https://github.com/CellProfiler/python-javabridge.git

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

"""
from pathlib import Path

import numpy as np
from rich.console import Console
console = Console()

import javabridge
import bioformats
from matplotlib import pyplot as plt

from skimage.filters import gaussian, laplace
from scipy import ndimage as ndi


def read_czi(path: Path) -> np.ndarray:
    """Read czi_data."""

    javabridge.start_vm(class_path=bioformats.JARS)

    with bioformats.ImageReader(str(path)) as reader:
        data = reader.read()
        print(type(data))
        print(data.shape)

    # metadata = bioformats.get_omexml_metadata(czi_path)
    # console.print(metadata)

    javabridge.kill_vm()
    return data


def min_max_normalization(image: np.ndarray):
    return (image-np.min(image))/(np.max(image)-np.min(image))


class Image:
    """Create image and serialize."""


    def __init__(self, channel_names, data):
        pass


def histogram_normalization():
    """Remove outliers at high values & rescale histogram to 256 range.

    set lowest entries to zero entries to zero,
    remove max entries
    remove long tail of intensities
     input image is converted according to the conventions of img_as_float (Normalized
     first to values [-1.0 ; 1.0] or [0 ; 1.0] depending on dtype of input)

    """
    pass


def plot_histogram(data: np.ndarray):
    i = 0
    data_flat = data[..., i].flatten()
    print(data_flat.max())
    hist, bins = np.histogram(data_flat, bins=256) # , 256, [0,256])

    print(hist)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    bin_width = bins[1] - bins[0]
    # plt.plot(cdf_normalized, label="cdf")
    plt.bar(bins[:-1], hist*bin_width, bin_width, color="black")

    plt.xscale = "log"
    # plt.xlim([0, 256])
    plt.show()


def gauss_filter(image: np.ndarray, sigma=10) -> np.ndarray:
    """Smoothening for processing.

    The filter sigma must depend on resolution, i.e., must be relative to the
    real size.

    The choice of the variance/covariance-matrix of your gaussian filter is extremely application dependent. There is no 'right' answer. That is like asking what bandwidth should one choose for a filter. Again, it depends on your application. Typically, you want to choose a gaussian filter such that you are nulling out a considerable amount of high frequency components in your image. One thing you can do to get a good measure, is compute the 2D DFT of your image, and overlay its co-efficients with your 2D gaussian image. This will tell you what co-efficients are being heavily penalized.

    """
    return gaussian(image, sigma=sigma, channel_axis=None)

def laplace_filter(image: np.ndarray, ksize=20) -> np.ndarray:
    """Edge detection."""
    return laplace(image, ksize=ksize)

def watershed(image):
    """
    Next, we want to separate the two circles. We generate markers
    at the maxima of the distance to the background:
    :return:
    """
    distance = ndi.distance_transform_edt(image)
    return distance

def discretize(image):
    """Create given number of classes."""
    pass

def contours(image):
    from skimage import measure

    # Construct some test data
    x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    r = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(r, 0.8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)



def zonation_position():
    pass


def plot_image(data: np.ndarray, cmap: str = "gray"):
    """Plot the relevant channels."""
    plt.figure(figsize=(24, 8))

    # E-catharin
    plt.subplot(131)
    plt.imshow(data[:, :, 0], cmap=cmap, aspect="equal", alpha=1.0)
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis('off')
    plt.title("ecad")

    # CYP2E1
    plt.subplot(132)
    plt.imshow(data[:, :, 2], cmap=cmap)
    plt.axis('off')
    plt.title("cyp2e1")

    # Zonation
    plt.subplot(133)
    # plt.imshow(data[:, :, 2]/data[:, :, 0], cmap=cmap)
    plt.imshow((data[:, :, 2] - data[:, :, 0]), cmap=cmap)
    plt.axis('off')
    plt.title("cyp2e1/ecad")
    plt.show()


if __name__ == "__main__":
    from zonation import CZI_PATH
    # data = read_czi(CZI_PATH / "M1039R_4x5_2.czi")
    # data = read_czi(CZI_PATH / "M1056_3_4x5.czi")
    # data = read_czi(CZI_PATH / "MH0117_4x5_2.czi")
    data = read_czi(CZI_PATH / "RH0422_2_4x5.czi")
    # data = read_czi(CZI_PATH / "Test1.czi")
    plot_image(data, cmap="gray")
    plot_image(data, cmap="tab20c")
    # plot_histogram(data)
    data_gauss = gauss_filter(data, sigma=10)
    plot_image(data_gauss, cmap="tab20c")
    # data_log = laplace_filter(data, ksize=100)
    # plot_image(data_log, cmap="tab20c")
    plot_histogram(data_gauss)
    # data_distance = watershed(data_gauss)
    # plot_image(data_distance)
