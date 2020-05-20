# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")

def plotFitsImage(image, wcs, filename):
    """
    Plot a 2D image of a nd numpy array
    """

    fig = plt.figure()
    fig.add_subplot(111, projection=wcs)

    plt.imshow(image, origin="lower", cmap="magma")

    cbar = plt.colorbar()
    cbar.set_label("Intensity")

    plt.title(f"{filename}")
    plt.xlabel("RA")
    plt.ylabel("Dec")

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

            # Get the world coordinate system data from the header
            wcs = WCS(file[0].header, naxis = 2)

            # Add the data to the list
            data.append([image_data, header_data, wcs])

    return data

def printFitsData(filenames, data):

    print("\n----------------------------------------------------------------------------")
    print("Data extracted from FITS files\n")

    for filename, data in zip(filenames, data):
        print(f"File: {filename}\n{data[0].shape[0]} x {data[0].shape[1]} pixels image")

        for header, value in data[1].items():
            print(f"{header}: {value}")

        print(f"\n{data[2]}\n")

    print("----------------------------------------------------------------------------")
