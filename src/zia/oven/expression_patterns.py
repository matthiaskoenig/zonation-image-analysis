import cv2
import numpy as np
import openslide
from matplotlib import pyplot as plt


PATH_TO_PIC = "/home/jkuettner/Pictures/wsi/J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.ndpi"

if __name__ == "__main__":
    image = openslide.OpenSlide(PATH_TO_PIC)

    x = 54064 - 1024
    y = 50547 - 1024

    region = image.read_region((x, y), 3, (1024, 1024))

    np_image = cv2.cvtColor(np.array(region), cv2.COLOR_BGR2HSV)
    threshold_value = 100, 150, 200
    max_value = 255
    h, s, v = cv2.split(np_image)

    for i in [h, s, v]:
        print(np.min(i), np.max(i))

    # Apply Gaussian blur to remove details
    blurred_s = cv2.GaussianBlur(s, (7, 7), 0)

    # Calculate the gradients using cv2.Sobel()
    gradient_x = cv2.Sobel(blurred_s, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_s, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude and direction of the gradients
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # _, threshold_v = cv2.threshold(dilated, 150, 255,
    #                              cv2.THRESH_BINARY)

    # hist = cv2.calcHist([b, g, r], [0, 0, 0], None, [256], [0, 256])

    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    ax: plt.Axes
    ax.imshow(s)
    ax1.imshow(blurred_s)
    ax2.imshow(gradient_magnitude)
    ax3.imshow(gradient_direction)
    plt.show()
