# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

from fitsFiles import *
from convolution import *
from radialProfileEllipse import *

if __name__ == "__main__":

    degreesToRadian = (np.pi / 180)
    HD100546_i = 42.46 * degreesToRadian #radian
    HD100546_PA = 139.1 * degreesToRadian #radian

    ALMA_filenames = os.listdir(ALMA_directory)
    relevant_headers = ["BMAJ", "BMIN", "BPA", "CDELT2"]
    descriptions = ["beamSemiMajor", "beamSemiMinor", "beamPA", "degreesPixelScale"]

    ALMA_data = getFitsData(ALMA_filenames, ALMA_directory, relevant_headers, descriptions)
    printFitsData(ALMA_filenames, ALMA_data)

    file_index = 0

    data = ALMA_data[file_index]
    image = data[0]
    header = data[1]
    wcs = data[2]

    pixelScale = abs(header["degreesPixelScale"]) # degrees per pixel
    major_axis = (header["beamSemiMajor"] * 2) / pixelScale # pixels
    minor_axis = (header["beamSemiMinor"] * 2) / pixelScale # pixels
    convolve_angle = header['beamPA'] * degreesToRadian # radian

    convolved_image = convolveImageGaussian2D(image, major_axis, minor_axis, convolve_angle)

    plotFitsImage(image, wcs, f"Original image ({ALMA_filenames[file_index]})")
    plotFitsImage(convolved_image, wcs, f"Convolved image ({ALMA_filenames[file_index]})\nmajor:{major_axis:.2f} px, minor:{minor_axis:.2f} px, angle:{(convolve_angle / np.pi):.2f} πrad")
    #
    eccentricity = np.sin(HD100546_i)

    plotEllipseProfile(convolved_image, eccentricity, HD100546_PA - (np.pi / 2))

    plt.show()
