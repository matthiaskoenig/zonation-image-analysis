"""Plotting helpers for images."""

from matplotlib import pyplot as plt
import numpy as np


def plot_image_with_hist(image: np.ndarray, cmap: str = "gray", title: str = None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap=cmap, aspect="equal")
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis('off')
    plt.title("CYP2E1")

    plt.subplot(223)
    hist, bins = np.histogram(image[:, :, 0], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("CYP2E1 Histogram")
    print(hist)

    # ECAD
    print("ECAD")
    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("E-Cadherin")

    plt.subplot(224)
    hist, bins = np.histogram(image[:, :, 1], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("E-Cadherin Histogram")
    print(hist)

    # Zonation
    # plt.subplot(133)
    # # plt.imshow(data[:, :, 2]/data[:, :, 0], cmap=cmap)
    # plt.imshow((data[:, :, 2] - data[:, :, 0]), cmap=cmap)
    # plt.axis('off')
    # plt.title("cyp2e1/ecad")
    plt.show()


def plot_zonation(image: np.ndarray, cmap="gray", title: str = None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1")

    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.axis('off')
    plt.title("E-Cadherin")

    # ECAD
    print("CYP2E1-ECAD")
    plt.subplot(223)
    plt.imshow(image[:, :, 0] - image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1 - E-Cadherin")

    print("CYP2E1/ECAD")
    plt.subplot(224)
    plt.imshow(image[:, :, 0]/image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1/E-Cadherin")

    plt.show()


def plot_overlay(image: np.ndarray, image_gauss, alpha=0.5, title: str = None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(image_gauss[:, :, 0] - image_gauss[:, :, 1], cmap="tab20c", aspect="equal", alpha=alpha)
    plt.axis('off')
    plt.title("CYP2E1 + Difference Overlay")

    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(image_gauss[:, :, 0] - image_gauss[:, :, 1], cmap="tab20c", aspect="equal", alpha=alpha)
    plt.axis('off')
    plt.title("E-Cadherin + Difference Overlay")

    plt.subplot(223)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(image_gauss[:, :, 0]/image_gauss[:, :, 1], cmap="tab20c", aspect="equal",
               alpha=alpha)
    plt.axis('off')
    plt.title("CYP2E1 + Ratio Overlay")

    plt.subplot(224)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(image_gauss[:, :, 0]/image_gauss[:, :, 1], cmap="tab20c", aspect="equal",
               alpha=alpha)
    plt.axis('off')
    plt.title("E-Cadherin + Ratio Overlay")

    plt.show()

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