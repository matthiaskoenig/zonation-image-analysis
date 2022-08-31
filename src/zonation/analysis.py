import numpy as np
from skimage.filters import gaussian, laplace
from scipy import ndimage as ndi

def zonation_position():
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


def min_max_normalization(image: np.ndarray):
    return (image-np.min(image))/(np.max(image)-np.min(image))



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


if __name__ == "__main__":
    from zonation import IMAGE_PATH
    from read_images import FluorescenceImage, Fluorophor
    from plots import plot_zonation

    # plot image channels & histograms
    fimage = FluorescenceImage.from_file(IMAGE_PATH / "Test1.pickle")
    cyp2e1 = fimage.get_channel_data(Fluorophor.)
    image = np.concatenate()
    plot_zonation()
