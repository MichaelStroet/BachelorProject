# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from equationsparameters import thermalIntensity

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = root_directory + "\\data\\"

def getImageMatrix(parameters, R_in, R_out, pixelDimension):
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
                matrix[i, j] = thermalIntensity(radius, parameters)

    return matrix

def plotImage(image, pixelDimension, pixelSize):
    """
    Plot a 2D image of the thermal continuum of the disk.
    """

    R_outer = (pixelDimension * pixelSize) / 2

    # Replace all zeros with the smallest value in the image
    smallest_value = np.min(image[np.nonzero(image)])
    image[np.where(image == 0.0)] = smallest_value
    print(image)

    plt.imshow(image, cmap="inferno", norm=LogNorm(), extent = [-R_outer, R_outer, -R_outer, R_outer])

    plt.title("Thermal Continuum disk")
    plt.xlabel("X [AU]")
    plt.ylabel("Y [AU]")

    cbar = plt.colorbar()
    cbar.set_label("Intensity [Jy/beam]")

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
    v = 365.5e9 # Hz
    R0 = 7 # AU
    T0 = 27 # K
    q0 = 2.6
    q1 = 0.26
    k = 0.34 # m^2 kg^-1 (at 365.5 GHz)
    Sig0 = 0.1 # kg m^-2 (guess)
    R_br = 47 # AU
    p0 = 0.53
    p1 = 8.0
    i = 0.0*np.pi # [0, np.pi/2]

    parameters = (v, R0, T0, q0, q1, Sig0, R_br, p0, p1, k, i)

    # Parameters
    R_inner = 1 # AU
    R_outer = 80 # AU
    pixelDimension = 500

    image = getImageMatrix(parameters, R_inner, R_outer, pixelDimension)
    saveImagePNG(image, pixelDimension, R_outer, f"codeFigures\\{pyName()}.png")

    plt.show()
