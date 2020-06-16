# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

from mcmcpool import mcmc
from fitsFiles import *

# Prevent unwanted use of multiple CPU threads
os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":

    degreesToRadian = (np.pi / 180)
    HD100546_i = 42.46 * degreesToRadian #radian
    HD100546_PA = 139.1 * degreesToRadian #radian

    ALMA_filenames = os.listdir(ALMA_directory)
    relevant_headers = ["BMAJ", "BMIN", "BPA", "CRPIX1", "CRPIX2", "CDELT2"]
    descriptions = ["beamSemiMajor", "beamSemiMinor", "beamPA", "xCenterPixel", "yCenterPixel", "degreesPixelScale"]

    ALMA_data = getFitsData(ALMA_filenames, ALMA_directory, relevant_headers, descriptions)
    printFitsData(ALMA_filenames, ALMA_data)

    file_index = 0
    data = ALMA_data[file_index]

    # Set negative noise from the image to zero
    data[0][np.where(data[0] < 0.0)] = 0.0

    # Add the inclination and corrected position angle to the header
    data[1]["inclination"] = HD100546_i
    data[1]["positionAngleMin90"] = HD100546_PA - (90 * degreesToRadian)

    mcmc(data)

    plt.show()
