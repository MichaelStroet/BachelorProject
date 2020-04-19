# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from equations import thermalIntensity

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = root_directory + "\\data\\"

def getImageMatrix(v, inc, R_in, R_out, pixelDimension):
    """
    Generates a numpy matrix of continuum intensities for plotting the disk.
    v is given in Hz, inc in radians [0, π/2] and R_in/R_out in AU.
    """

    coords = np.linspace(-R_out, R_out, pixelDimension)
    matrix = np.zeros((len(coords), len(coords)))

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            radius = np.sqrt(x**2 + y**2)
            if radius >= R_in and radius <= R_out:
                matrix[i, j] = thermalIntensity(v, radius, inc)

    return matrix

def plotImage(image, pixelDimension, pixelSize):
    """
    Plot a 2D image of the thermal continuum of the disk.
    """

    R_outer = (pixelDimension * pixelSize) / 2

    plt.imshow(image, cmap="inferno", extent = [-R_outer, R_outer, -R_outer, R_outer])

    plt.title("Thermal Continuum disk")
    plt.xlabel("X [AU]")
    plt.ylabel("Y [AU]")

    cbar = plt.colorbar()
    cbar.set_label("Intensity")

def saveImagePNG(image, pixelDimension, R_outer, filename):
    """
    Saves the image as a png file.
    """

    # Get the size per pixel in AU
    pixelSize = (2*R_outer) / pixelDimension

    plotImage(image, pixelDimension, pixelSize)
    plt.savefig(data_directory + filename)

def saveImageTXT(image, pixelDimension, R_outer, filename):
    """
    Saves the image as a txt file of the numpy array.
    """

    # Get the size per pixel in AU
    pixelSize = (2*R_outer) / pixelDimension

    # Save the image as a txt file with the pixelDimension and pixelSize in the header
    np.savetxt(data_directory + filename, image, fmt = "%.5e", header = f"{pixelDimension}, {pixelSize}")

def loadImageTXT(filename):
    """
    Loads an image saved as a txt file.
    """

    # Open the requested file and extract the header data and the image
    with open(data_directory + filename) as file:
        header_data = file.readline().replace(" ", "").strip("#\n").split(",")
        image = np.loadtxt(data_directory + filename)

    return image, (int(header_data[0]), float(header_data[1]))

if __name__ == "__main__":

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    R_inner = 1 # AU
    R_outer = 80 # AU
    pixelDimension = 500

    image = getImageMatrix(frequency, inclination, R_inner, R_outer, pixelDimension)
    saveImagePNG(image, pixelDimension, R_outer, f"codeFigures\\{pyName()}.png")

    plt.show()
