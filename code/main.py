# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add a path to the data folder
sys.path.append(os.path.join(root_directory, "data"))

from equations import *
from modelImage import *

if __name__ == "__main__":
    
    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    R_inner = 1 # AU
    R_outer = 200 # AU

    image = getImageMatrix(frequency, inclination, R_inner, R_outer)
    plotImage(image)
    plt.show()
