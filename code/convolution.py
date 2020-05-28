# Michael Stroet  11293284

import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import convolve, Gaussian2DKernel

def convolveImageGaussian2D(image, major, minor, theta, plot = False):
    """
    Convolve the image with a 2D Gaussian kernel.
    image is a nd numpy array,
    major and minor are the lengths of the axes of the FWHM ellipse in pixels,
    theta is the rotation angle in radians.
    """

    FWHM_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_x = major / FWHM_factor
    sigma_y = minor / FWHM_factor

    kernel = Gaussian2DKernel(sigma_x, sigma_y, theta)
    convolved_image = convolve(image, kernel)

    if plot:
        plotConvolution(image, kernel, convolved_image)

    return convolved_image

def plotConvolution(image, kernel, convolved):

    plt.figure("image convolution", figsize=(15, 6))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, origin="lower", cmap="inferno")
    ax1.set_title("Original")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(kernel, origin="lower", cmap="viridis")
    ax2.set_title(f"Kernel")
    ax2.set_xlabel("X [pixels]")
    ax2.set_ylabel("Y [pixels]")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(convolved, origin="lower", cmap="inferno")
    ax3.set_title("Convolved")
    ax3.set_xlabel("X [pixels]")
    ax3.set_ylabel("Y [pixels]")
