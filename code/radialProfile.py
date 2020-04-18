# Michael Stroet  11293284

# Extracts a radial profile from a 2D image

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from astropy.convolution import convolve, Gaussian2DKernel

from modelImage import loadImageTXT

def getLineCoordinates(pixelDimension, angle):
    """
    """

    center = (pixelDimension - 1) / 2.0
    edge = [center + center * np.cos(angle), center + center * np.sin(angle)]

    return center, edge

def getRadialProfile(image, pixelDimension, pixelSize, angle):
    """
    """

    center, edge = getLineCoordinates(pixelDimension, angle)
    N_points = pixelDimension

    # Generate N coordinates between the center and edge points
    x, y = np.linspace(center, edge[0], N_points), np.linspace(center, edge[1], N_points)

    # Get the radii of the coordinates
    radii = []
    for i in range(len(x)):
        radii.append((np.sqrt((x[i] - center)**2 + (y[i] - center)**2)) * pixelSize)

    # Extract the values along the line, using cubic interpolation
    image_profile = map_coordinates(image, np.vstack((x,y)))

    return radii, image_profile

def plotRadialProfileWithImage(image, pixelDimension, pixelSize, angle):
    """
    """

    radii, image_profile = getRadialProfile(image, pixelDimension, pixelSize, angle)
    center, edge = getLineCoordinates(pixelDimension, angle)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 7]}, figsize = (15, 6))

    ax1.imshow(image, cmap="inferno")
    ax1.plot([center, edge[0]], [center, edge[1]], "go-")
    ax1.set_title("Image")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2.plot(image_profile)
    ax2.set_title("Intensity radial profile")
    ax2.set_xlabel("R [pixels]")
    ax2.set_ylabel("Intensity")

if __name__ == "__main__":

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    profile_angle = (7/4) * np.pi
    kernel_angle = (1/3) * np.pi

    image, (pixelDimension, pixelSize) = loadImageTXT("image.txt")
    R_outer = pixelDimension*pixelSize

    image_radii, image_profile = getRadialProfile(image, pixelDimension, pixelSize, profile_angle)

    kernel = Gaussian2DKernel(x_stddev = 10, y_stddev = 8, theta = kernel_angle)
    convolved_image = convolve(image, kernel)
    convolved_radii, convolved_profile = getRadialProfile(convolved_image, pixelDimension, pixelSize, profile_angle)

    center, edge = getLineCoordinates(pixelDimension, profile_angle)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 7]}, figsize = (15, 12))

    ax1.imshow(image, cmap="inferno") #, extent = [-R_outer, R_outer, -R_outer, R_outer])
    ax1.plot([center, edge[0]], [center, edge[1]], "go-")
    ax1.set_title("Original")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2.plot(image_radii, image_profile)
    ax2.set_title("Intensity radial profile")
    ax2.set_xlabel("Radius [AU]")
    ax2.set_ylabel("Intensity")

    ax3.imshow(convolved_image, cmap="inferno")
    ax3.plot([center, edge[0]], [center, edge[1]], "go-")
    ax3.set_title("Convolved")
    ax3.set_xlabel("X [pixels]")
    ax3.set_ylabel("Y [pixels]")

    ax4.plot(convolved_radii, convolved_profile)
    ax4.set_title("Intensity radial profile")
    ax4.set_xlabel("Radius [AU]")
    ax4.set_ylabel("Intensity")

    plt.savefig(figures_directory + pyName() + ".png")

    # plotRadialProfileWithImage(image, pixelDimension, pixelSize, profile_angle)
    # plotRadialProfileWithImage(convolved_image, pixelDimension, pixelSize, profile_angle)

    plt.show()
