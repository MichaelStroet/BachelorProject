# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from equations import thermalIntensity

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = root_directory + "\\data\\"

def getImageMatrix(v, inc, R_in, R_out):
    """
    Generates a numpy matrix of continuum intensities for plotting the disk.
    v is given in Hz, inc in radians [0, π/2] and R_in/R_out in AU.
    """

    coords = np.linspace(-R_out, R_out, 1000)
    matrix = np.zeros((len(coords), len(coords)))

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            radius = np.sqrt(x**2 + y**2)
            if radius >= R_in and radius <= R_out:
                matrix[i, j] = thermalIntensity(v, radius, inc)

    return matrix

def plotImage(image):
    """
    Plot a 2D image of the thermal continuum of the disk.
    """

    plt.imshow(image, cmap="inferno")

    plt.title("Thermal Continuum disk")
    plt.xlabel("X [AU]")
    plt.ylabel("Y [AU]")

    cbar = plt.colorbar()
    cbar.set_label("Intensity")

def saveImagePNG(image, filename):
    """
    Saves the image as a png file.
    """
    plotImage(image)
    plt.savefig(data_directory + filename)

def saveImageTXT(image, filename):
    """
    Saves the image as a txt file of the numpy array.
    """

    np.savetxt(data_directory + filename, image)

def loadImageTXT(filename):
    """
    Loads an image saved as a txt file.
    """

    image = np.loadtxt(data_directory + filename)
    return image

if __name__ == "__main__":

    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    R_inner = 1 # AU
    R_outer = 200 # AU

    image = getImageMatrix(frequency, inclination, R_inner, R_outer)
    plotImage(image)
    plt.show()
