# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")

def plotFitsImage(data, filename, coordinates = False):
    """
    Plot a 2D image of a nd numpy array
    """
    image = np.copy(data[0])
    header, wcs = data[1:]

    # Replace all negatives with the smallest positive value in the image
    smallest_value = np.min(image[np.where(image > 0)])
    image[np.where(image <= 0.0)] = smallest_value

    fig = plt.figure()

    if coordinates:
        fig.add_subplot(111, projection = wcs)

        plt.imshow(image, origin="lower", norm=LogNorm(), cmap="inferno")

        plt.xlabel("RA")
        plt.ylabel("Dec")

    else:
        centerPixel = (header["xCenterPixel"], header["yCenterPixel"])
        pixelDimension = image.shape

        degreesToArcseconds = 3600
        pixelScale = header["degreesPixelScale"] * degreesToArcseconds

        extent = [(-centerPixel[0]) * pixelScale, (pixelDimension[0] - centerPixel[0]) * pixelScale,
            (-centerPixel[1]) * pixelScale, (pixelDimension[1] - centerPixel[1]) * pixelScale]

        # print(f"centerPixel: {centerPixel}")
        # print(f"pixelDimension: {pixelDimension}")
        # print(f"extent: {extent}")

        plt.imshow(image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)

        plt.xlabel("Arcseconds")
        plt.ylabel("Arcseconds")

    cbar = plt.colorbar()
    cbar.set_label("Intensity [Jy/beam]")

    plt.title(f"{filename}")

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
