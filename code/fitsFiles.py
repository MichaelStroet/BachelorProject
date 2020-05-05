# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")

def plotFitsImage(image, filename):
    """
    Plot a 2D image of a nd numpy array
    """

    plt.figure()

    plt.imshow(image, cmap="inferno")

    cbar = plt.colorbar()
    cbar.set_label("Intensity")

    plt.title(f"{filename}")
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")

def getFitsData(filenames, directory_path, headers, descriptions):

    data = []

    # Open the Fits files and extract the relevant data
    for filename in filenames:
        with fits.open(os.path.join(directory_path, filename)) as file:

            # Get the image data
            image_data = file[0].data[0][0]
            image_data[np.isnan(image_data)] = 0

            # Get the relevant header data
            header_data = {}
            for header_name, description in zip(headers, descriptions):
                header_data[description] = file[0].header[header_name]

            # Add the data to the list
            data.append([image_data, header_data])

    return data

def printFitsData(filenames, data):

    print("Data extracted from FITS files\n")

    for filename, data in zip(filenames, data):
        print(f"File: {filename}\n{data[0].shape[0]} x {data[0].shape[1]} pixels image")

        for header, value in data[1].items():
            print(f"{header}: {value}")

        print()
