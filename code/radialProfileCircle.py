# Michael Stroet  11293284

# Extracts a radial profile from a 2D image

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from astropy.convolution import convolve, Gaussian2DKernel

from modelImage import loadImageTXT, plotImage
from convolution import convolveImageGaussian2D

N_RADII = 1000
N_ANGLES = 1000

def getCircleCoordinates(center, R):
    """
    Returns coordinates for a circle of radius 'R' centered at 'center'
    """

    angles = np.linspace(0, 2*np.pi, N_ANGLES)

    coordinates = [[], []]
    for angle in angles:
        coordinates[0].append(center + R * np.cos(angle))
        coordinates[1].append(center + R * np.sin(angle))

    return coordinates

def getCircleAverage(image, R):
    """
    Determines the average value of points on a circle of radius 'R' centered at 'center'
    'R' and 'center' are given in pixel values.
    """

    center = (image.shape[0] - 1) / 2.
    x_coords, y_coords = getCircleCoordinates(center, R)

    # Extract the values along the circle, using cubic interpolation
    values = map_coordinates(image, [x_coords, y_coords])
    average = sum(values) / len(values)

    return average


def getCircleProfile(image, R_outer):
    """
    Creates a radial profile of an image by averaging values of circles.
    'R_outer' is given in pixel values.
    """

    radii = np.linspace(0, R_outer, N_RADII)

    radial_profile = []
    for radius in radii:
        radial_profile.append(getCircleAverage(image, radius))

    return radial_profile


def plotCircleProfile(image, pixelSize):

    R_outer = image.shape[0] / 2

    radial_profile = getCircleProfile(image, R_outer)
    radii = np.linspace(0, R_outer, N_RADII)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 7]}, figsize = (15, 6))

    test = ax1.imshow(image, cmap="inferno")
    plt.colorbar(test, ax = ax1, orientation = "vertical")

    ax1.set_title("Image")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2.plot(radii, radial_profile)
    ax2.set_title("Intensity radial profile (circles)")
    ax2.set_xlabel("R [pixels]")
    ax2.set_ylabel("Intensity")


if __name__ == "__main__":

    import os, sys

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    sigma_x = 10
    sigma_y = 6
    kernel_angle = (2/3) * np.pi

    image, (pixelDimension, pixelSize) = loadImageTXT("image.txt")
    convolved_image = convolveImageGaussian2D(image, sigma_x, sigma_y, kernel_angle)

    plotCircleProfile(image, pixelSize)
    plotCircleProfile(convolved_image, pixelSize)

    plt.show()
