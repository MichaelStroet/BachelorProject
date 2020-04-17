# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import convolve, Gaussian2DKernel

from modelImage import *

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = root_directory + "\\data\\"
figures_directory = data_directory + "codeFigures\\"

def pyName():
    return __file__.split("\\")[-1].replace(".py", "")

if __name__ == "__main__":

    image, pixelDimension, pixelSize = loadImageTXT("image.txt")
    kernel = Gaussian2DKernel(x_stddev = 10, y_stddev = 8, theta = np.pi/3)

    convolved_image = convolve(image, kernel)

    plt.figure(1, figsize=(12, 6)).clf()

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap="inferno")
    ax1.set_title("Original")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(kernel, interpolation="none")
    ax2.set_title("Kernel")
    ax2.set_xlabel("X [pixels]")
    ax2.set_ylabel("Y [pixels]")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(convolved_image, cmap="inferno")
    ax3.set_title("Convolved")
    ax3.set_xlabel("X [pixels]")
    ax3.set_ylabel("Y [pixels]")

    plt.savefig(figures_directory + pyName() + ".png")

    plt.show()
