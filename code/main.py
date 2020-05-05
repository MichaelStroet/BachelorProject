# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

from modelImage import *
from fitsFiles import *

from radialProfileEllipse import *

if __name__ == "__main__":

    HD100546_i = 42.46 * (np.pi / 180) #radian
    HD100546_PA = 139.1 * (np.pi / 180) #radian

    ALMA_filenames = os.listdir(ALMA_directory)
    relevant_headers = ["BMAJ", "BMIN", "BPA", "CDELT1"]
    descriptions = ["beamMajor", "beamMinor", "beamPA", "pixelScale"]

    ALMA_data = getFitsData(ALMA_filenames, ALMA_directory, relevant_headers, descriptions)
    printFitsData(ALMA_filenames, ALMA_data)

    data = ALMA_data[0]
    image = data[0]

    eccentricity = 0.8
    rotation = (-1/3) * np.pi

    semi_major = 25
    plotEllipseImage(image, semi_major, eccentricity, rotation)
    plotEllipseProfile(image, eccentricity, rotation)

    plt.show()
