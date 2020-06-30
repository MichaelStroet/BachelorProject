# Michael Stroet  11293284

import os, sys
import numpy as np

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")

def cropImage(img, crop_radius):
   y, x = img.shape
   startx = x // 2 - crop_radius
   starty = y // 2 - crop_radius
   return img[starty:starty + 2*crop_radius, startx:startx + 2*crop_radius]

def getImageMatrix(fixed_pars, free_pars, pixel_coords, arcsec_per_pix, sr_per_pix, model):
    """
    Generates a numpy matrix of continuum intensities for plotting the disk.
    """
    # Import correct thermal intensity model
    if model == "TWHya":
        from equations import thermalIntensityTWHya as thermalIntensity

    elif model == "single":
        from equations import thermalIntensitySingle as thermalIntensity

        Rin = free_pars[0]
        Rout = free_pars[1]

    elif model == "double":
        from equations import thermalIntensityDouble as thermalIntensity

        Rin = fixed_pars[6]
        Rout = free_pars[0]

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

if __name__ == "__main__":

    from convolution import *
    from fitsFiles import *

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from astropy.convolution import convolve
    from radialProfileCircle import getCircleProfile

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

    degreesToRadian = (np.pi / 180)
    HD100546_i = 42.46 * degreesToRadian #radian
    HD100546_PA = 139.1 * degreesToRadian #radian

    ALMA_filenames = os.listdir(ALMA_directory)
    relevant_headers = ["BMAJ", "BMIN", "BPA", "CRPIX1", "CRPIX2", "CDELT2"]
    descriptions = ["beamSemiMajor", "beamSemiMinor", "beamPA", "xCenterPixel", "yCenterPixel", "degreesPixelScale"]

    ALMA_data = getFitsData(ALMA_filenames, ALMA_directory, relevant_headers, descriptions)
    # printFitsData(ALMA_filenames, ALMA_data)

    file_index = 1
    data = ALMA_data[file_index]

    # Set negative noise from the image to zero
    data[0][np.where(data[0] < 0.0)] = 0.0

    # Add the inclination and corrected position angle to the header
    data[1]["inclination"] = HD100546_i
    data[1]["positionAngleMin90"] = HD100546_PA - (90 * degreesToRadian)

    inclination = data[1]["inclination"]
    eccentricity = np.sin(inclination)
    rotation = data[1]["positionAngleMin90"]

    total_intensity_radii = 250

    pixel_dimension = min(data[0].shape) # pixels
    pixel_radius = pixel_dimension / 2 # pixels
    pixel_coords = np.linspace(-pixel_radius, pixel_radius, pixel_dimension) # pixels
    pixel_radii = np.linspace(0, pixel_radius, total_intensity_radii) # pixels

    arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
    sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    arcsec_radius = pixel_radius * arcsec_per_pix
    arcsec_radii = np.linspace(0, arcsec_radius, total_intensity_radii)

    model = "single"

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = inclination # radian
    T0 = 30 # K
    q = 0.25
    Sig0 = 0.25 # kg m^-2

    # Free parameters guesses
    Rin = 0.11 # Arcseconds
    Rout = 0.89  # Arcseconds
    p = 0.81

    fixed_pars = (v, k, i, T0, q, Sig0)
    free_pars = np.array([Rin, Rout, p])

    # Generate the convolution kernels for the model image
    model_kernel_area, model_kernel_peak = generateModelKernels(data)
    model_kernel = model_kernel_area

    model_image = getImageMatrix(fixed_pars, free_pars, pixel_coords, arcsec_per_pix, sr_per_pix, model)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])
    model_intensities = np.asarray(getCircleProfile(model_image, pixel_radii))

    convolved_image = convolve(model_image, model_kernel)
    convolved_image[np.where(convolved_image <= 0.0)] = np.min(convolved_image[np.where(convolved_image > 0)])
    convolved_intensities = np.asarray(getCircleProfile(convolved_image, pixel_radii))

    centerPixel = (data[1]["xCenterPixel"], data[1]["yCenterPixel"])
    pixelDimension = data[0].shape

    extent = [(-centerPixel[0]) * arcsec_per_pix, (pixelDimension[0] - centerPixel[0]) * arcsec_per_pix,
        (-centerPixel[1]) * arcsec_per_pix, (pixelDimension[1] - centerPixel[1]) * arcsec_per_pix]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    ax1.imshow(model_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax1.set_title("Original model")

    ax2.plot(arcsec_radii, model_intensities)
    ax2.set_title("Intensity profile")

    ax3.imshow(convolved_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax3.set_title("Convolved model")

    ax4.plot(arcsec_radii, convolved_intensities)
    ax4.set_title("Intensity profile")

    plt.show()
