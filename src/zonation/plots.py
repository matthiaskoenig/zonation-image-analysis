from matplotlib import plt
from matplotlib import pyplot as plt


def plot_zonation(data: np.ndarray, cmap: str = "gray"):
    """Plot the relevant channels."""
    plt.figure(figsize=(24, 8))

    # CYP2E1
    plt.subplot(131)
    plt.imshow(data[:, :, 0], cmap=cmap, aspect="equal", alpha=1.0)
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis('off')
    plt.title("CYP2E1")

    # CYP2E1
    plt.subplot(132)
    plt.imshow(data[:, :, 2], cmap=cmap)
    plt.axis('off')
    plt.title("E-Cadherin")

    # Zonation
    # plt.subplot(133)
    # # plt.imshow(data[:, :, 2]/data[:, :, 0], cmap=cmap)
    # plt.imshow((data[:, :, 2] - data[:, :, 0]), cmap=cmap)
    # plt.axis('off')
    # plt.title("cyp2e1/ecad")
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