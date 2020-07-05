# Michael Stroet  11293284

import os, sys
import numpy as np

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")

def cropImage(image, crop_radius):
   y, x = image.shape
   startx = x // 2 - crop_radius
   starty = y // 2 - crop_radius
   return image[starty:starty + 2*crop_radius, startx:startx + 2*crop_radius]

def getImageMatrix(fixed_pars, free_pars, pixel_coords, arcsec_per_pix, sr_per_pix, model):
    """
    Generates a numpy matrix of continuum intensities for plotting the disk.
    """
    # Import correct thermal intensity model
    if model == "single":
        from equations import thermalIntensitySingle as thermalIntensity

        Rin = free_pars[0]
        Rout = free_pars[1]

    elif model == "double":
        from equations import thermalIntensityDouble as thermalIntensity

        Rin = fixed_pars[6]
        Rout = free_pars[0]

    elif model == "gaussian":
        from equations import thermalIntensityGaussian as thermalIntensity

        Rin = free_pars[0]
        Rout = free_pars[1]

    else:
        print(f"Error: Unknown model {model}")
        exit(1)

    matrix = np.zeros((len(pixel_coords), len(pixel_coords)))

    for i, x in enumerate(pixel_coords):
        for j, y in enumerate(pixel_coords):
            radius = np.sqrt(x**2 + y**2) * arcsec_per_pix
            if radius >= Rin and radius <= Rout:
                matrix[i, j] = thermalIntensity(radius, sr_per_pix, fixed_pars, free_pars)

    return matrix

def plotImage(image, Rout, plot_title = ""):
    """
    Plot a 2D image of the thermal continuum of the disk.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Replace all zeros with the smallest value in the image
    smallest_value = np.min(image[np.nonzero(image)])
    image[np.where(image == 0.0)] = smallest_value

    plt.figure()

    plt.imshow(image, cmap="inferno", norm=LogNorm(), extent = [-Rout, Rout, -Rout, Rout])

    plt.title(plot_title)
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")

    cbar = plt.colorbar()
    cbar.set_label("Intensity [Jy/beam]")

def saveImagePNG(image, pixel_dimension, R_outer, filename):
    """
    Saves the image as a png file.
    """

    # Get the size per pixel in AU
    pixel_scale = (2*R_outer) / pixel_dimension

    plotImage(image, R_outer)
    plt.savefig(data_directory + filename)

def saveImageTXT(image, pixel_dimension, R_outer, filename):
    """
    Saves the image as a txt file of the numpy array.
    """

    # Get the size per pixel in AU
    pixel_scale = (2*R_outer) / pixel_dimension

    # Save the image as a txt file with the pixel_dimension and pixelSize in the header
    np.savetxt(data_directory + filename, image, fmt = "%.5e", header = f"{pixel_dimension}, {pixel_scale}")

def loadImageTXT(filename):
    """
    Loads an image saved as a txt file.
    """

    # Open the requested file and extract the header data and the image
    with open(data_directory + filename) as file:
        header_data = file.readline().replace(" ", "").strip("#\n").split(",")
        image = np.loadtxt(data_directory + filename)

    return image, (int(header_data[0]), float(header_data[1]))
