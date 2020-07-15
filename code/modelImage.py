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
