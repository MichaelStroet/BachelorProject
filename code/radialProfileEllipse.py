# Michael Stroet  11293284

# Extracts a radial profile from a 2D image

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from astropy.convolution import convolve, Gaussian2DKernel

from modelImage import loadImageTXT, plotImage
from convolution import convolveImageGaussian2D

N_RADII = 100
N_ANGLES = 1000

def getUnits(pixelSize):
    if pixelSize == 1:
        return "pixels"
    else:
        return "also pixels"

def getEllipseCoordinates(center, major, ecc, theta):
    """
    Returns coordinates for a ellipse with eccentricity 'ecc'
    and semi-major axis 'major' centered at 'center' and rotated by angle 'theta'
    """

    angles = np.linspace(0, 2*np.pi, N_ANGLES)
    minor = major * np.sqrt(1 - ecc**2)

    coordinates = [[], []]
    for angle in angles:
        coordinates[0].append(center[0] + (major * np.cos(angle) * np.cos(theta)) - (minor * np.sin(angle) * np.sin(theta)))
        coordinates[1].append(center[1] + (major * np.cos(angle) * np.sin(theta)) + (minor * np.sin(angle) * np.cos(theta)))

    return coordinates

def getEllipseAverage(image, semi_major, eccentricity, rotation):
    """
    """

    center = [(image.shape[0] - 1) / 2., (image.shape[1] - 1) / 2.]
    coordinates = getEllipseCoordinates(center, semi_major, eccentricity, rotation)

    # Extract the values along the ellipse, using cubic interpolation
    values = map_coordinates(image, coordinates)
    average = sum(values) / len(values)

    return average

def getEllipseProfile(image, R_outer, eccentricity, rotation):
    """
    """

    major_axes = np.linspace(0, R_outer, N_RADII)
    radial_profile = []

    for semi_major in major_axes:
        radial_profile.append(getEllipseAverage(image, semi_major, eccentricity, rotation))

    return radial_profile

def plotEllipseImage(image, semi_major, eccentricity, rotation, pixelSize = 1):
    """
    Plots an ellips in the image
    """

    units = getUnits(pixelSize)

    center = [(image.shape[0] - 1) / 2., (image.shape[1] - 1) / 2.]
    x_coords, y_coords = getEllipseCoordinates(center, semi_major, eccentricity, rotation)

    plt.figure()

    plt.imshow(image, cmap="inferno")
    plt.colorbar()

    plt.plot(x_coords, y_coords)

    plt.xlabel(f"X [{units}]")
    plt.ylabel(f"Y [{units}]")

def plotEllipseProfile(image, eccentricity, rotation, pixelSize = 1):

    units = getUnits(pixelSize)

    R_outer = min(image.shape[0] / 2, image.shape[1] / 2)
    # R_outer = 30

    radial_profile = getEllipseProfile(image, R_outer, eccentricity, rotation)
    major_axes = np.linspace(0, R_outer, N_RADII)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 7]}, figsize = (15, 6))

    ax1.imshow(image, cmap="inferno")
    ax1.set_title("Image")
    ax1.set_xlabel(f"X [{units}]")
    ax1.set_ylabel(f"Y [{units}]")

    ax2.plot(major_axes, radial_profile)
    ax2.set_title("Intensity radial profile (ellipses)")
    ax2.set_xlabel(f"Semi-major axis [{units}]")
    ax2.set_ylabel("Intensity [Jy/beam]")


if __name__ == "__main__":

    import os, sys

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    image, (pixelDimension, pixelSize) = loadImageTXT("image.txt")

    semi_major = 130
    eccentricity = 0.9
    rotation = (1/3) * np.pi

    plotEllipseImage(image, semi_major, eccentricity, rotation, pixelSize)

    plotEllipseProfile(image, eccentricity, rotation, pixelSize)

    plt.show()
