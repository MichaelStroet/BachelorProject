# Michael Stroet  11293284

# Extracts a radial profile from a 2D image

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from astropy.convolution import convolve, Gaussian2DKernel

from modelImage import loadImageTXT

def getRadialProfile(image, pixelDimension, pixelSize, angle):
    pass

if __name__ == "__main__":



    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")
        
    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    image, (pixelDimension, pixelSize) = loadImageTXT("image.txt")
    print(pixelDimension, pixelSize)
    print(image)

    kernel = Gaussian2DKernel(x_stddev = 10, y_stddev = 8, theta = np.pi/3)
    convolved_image = convolve(image, kernel)

    center = pixelDimension / 2.0
    end = [pixelDimension, center]
    N_points = pixelDimension

    # Generate N coordinates between the center and end points
    x, y = np.linspace(center, end[0], N_points), np.linspace(center, end[1], N_points)

    # Get the radii of the coordinates
    radii = []
    for i in range(len(x)):
        radii.append((np.sqrt((x[i] - center)**2 + (y[i] - center)**2)) * pixelSize)
    print(radii)

    # Extract the values along the line, using cubic interpolation
    image_profile = map_coordinates(image, np.vstack((x,y)))
    convolved_profile = map_coordinates(convolved_image, np.vstack((x,y)))

    #-- Plot...
    plt.figure("Testing", figsize = (10, 10))

    ax1 = plt.subplot(2,2,1)

    ax1.imshow(image, cmap="inferno")
    ax1.plot([center, end[0]], [center, end[1]], "go-")
    ax1.set_title("Original")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2 = plt.subplot(2,2,2)

    ax2.imshow(convolved_image, cmap="inferno")
    ax2.plot([center, end[0]], [center, end[1]], "go-")
    ax2.set_title("Convolved")
    ax2.set_xlabel("X [pixels]")
    ax2.set_ylabel("Y [pixels]")

    ax3 = plt.subplot(2,2,3)
    ax3.plot(image_profile)

    ax4 = plt.subplot(2,2,4)
    ax4.plot(convolved_profile)

    plt.savefig(figures_directory + pyName() + ".png")

    plt.show()
