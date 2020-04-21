# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from fitsImage import *
from matplotlib.colors import LogNorm

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

from modelImage import *
from convolution import *

if __name__ == "__main__":

    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    R_inner = 1 # AU
    R_outer = 100 # AU
    pixelDimension = 500

    # image = getImageMatrix(frequency, inclination, R_inner, R_outer, pixelDimension)
    # saveImagePNG(image, pixelDimension, R_outer, f"image.png")
    # saveImageTXT(image, pixelDimension, R_outer, "image.txt")
    #
    # plt.show()

    ALMA_fits = os.listdir(ALMA_directory)

    with fits.open(os.path.join(ALMA_directory, ALMA_fits[1])) as fits_file:
        fits_file.info()

        image = fits_file[0].data[0][0]
        print()
        print(image)
        print(image.shape)
        print()

    sigma_x = 3
    sigma_y = 2
    theta = -(1/3) * np.pi

    # image, (pixelDimension, pixelSize) = loadImageTXT("image.txt")

    # print(f"Convolving image with sigx = {sigma_x}, sigy = {sigma_y}, theta = {theta}")
    # convolved_image = convolveImageGaussian2D(image, sigma_x, sigma_y, theta)
    # print("done\n")

    plotFitsImage(image)

    zerod_image = np.copy(image)
    zerod_image[np.isnan(zerod_image)] = 0

    profileRadius = 500

    # from astropy.convolution import Gaussian2DKernel
    # kernel = Gaussian2DKernel(sigma_x, sigma_y, -theta)
    #
    # plt.figure("2")
    # ax1 = plt.subplot(1, 3, 1)
    # ax1.imshow(image, cmap="inferno")
    # ax1.set_title("Original")
    # ax1.set_xlabel("X [pixels]")
    # ax1.set_ylabel("Y [pixels]")
    #
    # ax2 = plt.subplot(1, 3, 2)
    # ax2.imshow(kernel, cmap="viridis")
    # ax2.set_title(f"Kernel")
    # ax2.set_xlabel("X [pixels]")
    # ax2.set_ylabel("Y [pixels]")
    #
    # ax3 = plt.subplot(1, 3, 3)
    # ax3.imshow(convolved_image, cmap="inferno")
    # ax3.set_title("Convolved")
    # ax3.set_xlabel("X [pixels]")
    # ax3.set_ylabel("Y [pixels]")

    plotFitsRadialProfileWithImage(zerod_image, zerod_image.shape[0], profileRadius, theta)

    plt.show()
