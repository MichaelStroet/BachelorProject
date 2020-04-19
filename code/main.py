# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add a path to the data folder
sys.path.append(os.path.join(root_directory, "data"))

from modelImage import *

if __name__ == "__main__":

    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    R_inner = 1 # AU
    R_outer = 100 # AU
    pixelDimension = 500

    image = getImageMatrix(frequency, inclination, R_inner, R_outer, pixelDimension)
    saveImagePNG(image, pixelDimension, R_outer, f"image.png")
    saveImageTXT(image, pixelDimension, R_outer, "image.txt")

    plt.show()
