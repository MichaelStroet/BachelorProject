# Michael Stroet  11293284

# This script contains the equations used in the model
# Run the script itself to generate a plot for each equation

import numpy as np

### Planck black body intensity -------------------------------------------------------------------------------------------------

def planckFunction(v, T, sr_per_pix):
    """
    Returns the intensity of thermal radiation a black body emits
    at frequency v and temperature T.
    """

    # Constants
    h = 6.626e-34 # J s (Planck"s constant)
    c = 2.998e8 # m s^-1 (Speed of light)
    kB = 1.38e-23 # J K^-1 (Boltzmann"s constant)

    if T <= 0:
        # print(f"Nonsense temperature: {T}K\nReturn intensity of 1e-100\n")
        return 1e-100

    exponent = h * v / kB / T
    if exponent > 700:
        # print(f"Exponent too large to handle: e^{exponent:.2f}\nReturn intensity of 1e-100\n")
        return 1e-100
    #
    exp = np.exp(exponent) - 1
    if exp < 1e-300:
        # print(f"Too close to division by zero: 1/{exp}\nReturn intensity of 1e-100\n")
        return 1e-100

    # Calculate intensity in J/m^2/sr
    planck = 2 * h * v**3 / c**2 / exp

    # Convert J/m^2/sr to Jy/pixel
    return 1e26 * sr_per_pix * planck

### Dust surface density --------------------------------------------------------------------------------------------------------

# def dustSurfaceDensityTWHya(R, Sig0, R_br, p0, p1):
#     """
#     Calculates the dust surface density (Sigma d) from a 2-sloped power law.
#     """
#
#     if R <= R_br:
#         return Sig0 * pow(R / R_br, -p0)
#     else:
#         return Sig0 * pow(R / R_br, -p1)

def dustSurfaceDensitySingle(R, Rin, Sig0, p):
    """
    Calculates the dust surface density (Sigma d) from single power law.
    """

    return Sig0 * pow(R / Rin, -p)

def dustSurfaceDensityDouble(R, Sig1, Sig2, R1, p1, p2):
    """
    Calculates the dust surface density (Sigma d) from a 2-sloped discontinuous power law.
    """

    if R <= R1:
        return Sig1 * pow(R / R1, -p1)
    else:
        return Sig2 * pow(R / R1, -p2)

### Dust optical depth ----------------------------------------------------------------------------------------------------------

# def dustOpticalDepthTWHya(R, Sig0, R_br, p0, p1, k, i):
#     """
#     Calculates the dust optical depth (tau) for radius R.
#     """
#     return dustSurfaceDensityTWHya(R, Sig0, R_br, p0, p1) * k * np.cos(i)

def dustOpticalDepthSingle(R, Rin, Sig0, p, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensitySingle(R, Rin, Sig0, p) * k * np.cos(i)

def dustOpticalDepthDouble(R, Sig1, Sig2, R1, p1, p2, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensityDouble(R, Sig1, Sig2, R1, p1, p2) * k * np.cos(i)

### Disk temperature  -----------------------------------------------------------------------------------------------------------

# def diskTemperatureTWHya(R, R0, T0, q0, q1):
#     """
#     Calculates the temperature at radius R from a broken power law.
#     """
#
#     if R <= R0:
#         return T0 * pow(R / R0, -q0)
#     else:
#         return T0 * pow(R / R0, -q1)

def diskTemperature(R, Rin, T0, q):
    """
    Calculates the temperature at radius R from a single power law.
    """

    return T0 * pow(R / Rin, -q)

### Thermal continuum intensity -------------------------------------------------------------------------------------------------

# def thermalIntensityTWHya(R, sr_per_pix, fixed_pars, free_pars):
#     """
#     Calculates the thermal intensity from the dust at radius R.
#     """
#
#     v, k, i = fixed_pars
#     Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0 = free_pars
#
#     T_disk = diskTemperatureTWHya(R, R0, T0, q0, q1)
#     optical_depth = dustOpticalDepthTWHya(R, Sig0, R_br, p0, p1, k, i)
#
#     return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensitySingle(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, q, Sig0 = fixed_pars
    Rin, Rout, p = free_pars

    T_disk = diskTemperature(R, Rin, T0, q)
    optical_depth = dustOpticalDepthSingle(R, Rin, Sig0, p, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensityDouble(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, q, Sig1, Rin, R1, p1 = fixed_pars
    Rout, SigFrac, p2 = free_pars
    Sig2 = SigFrac * Sig1

    T_disk = diskTemperature(R, Rin, T0, q)
    optical_depth = dustOpticalDepthDouble(R, Sig1, Sig2, R1, p1, p2, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

if __name__ == "__main__":

    import os, sys
    import matplotlib.pyplot as plt

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_directory = os.path.join(root_directory, "data")
    ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

    from fitsFiles import *

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

    arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
    sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = HD100546_i # radian
    T0 = 30 # K
    Sig0 = 0.25 # kg m^-2

    Rin = 0.6
    Rout = 2
    p = 0.1
    q = 1.8

    fixed_pars = (v, k, i, T0, q, Sig0)
    free_pars = [Rin, Rout, p]

    total_intensity_radii = 100

    data_dimension = data[0].shape[0]
    data_radius = data_dimension / 2
    data_coords = np.linspace(-data_radius, data_radius, data_dimension)

    ### -------------------------------------------------------------------------------------------

    data_matrix = np.zeros((len(data_coords), len(data_coords)))

    for i, x in enumerate(data_coords):
        for j, y in enumerate(data_coords):
            radius = np.sqrt(x**2 + y**2) * arcsec_per_pix
            if radius >= Rin and radius <= Rout:
                data_matrix[i, j] = thermalIntensitySingle(radius, sr_per_pix, fixed_pars, free_pars)

    print(data_matrix)

    data_centerPixel = [data_dimension / 2, data_dimension / 2]
    data_extent = [(-data_centerPixel[0]) * arcsec_per_pix, (data_dimension - data_centerPixel[0]) * arcsec_per_pix,
        (-data_centerPixel[1]) * arcsec_per_pix, (data_dimension - data_centerPixel[1]) * arcsec_per_pix]

    plt.figure("data")
    plt.imshow(data_matrix, origin = "lower", extent = data_extent, cmap = "inferno")
    plt.xlim([-1,1])
    plt.ylim([-1,1])

    plt.title("'data'-scale model")
    plt.colorbar()
    ### -------------------------------------------------------------------------------------------

    model_scale = 3

    model_arcsec_per_pix = arcsec_per_pix / model_scale
    model_sr_per_pix = sr_per_pix / model_scale

    model_dimension = model_scale * data_dimension
    model_radius = model_scale * data_radius
    model_coords = np.linspace(-model_radius, model_radius, model_dimension)

    model_Rin = Rin * model_scale
    model_Rout = Rout * model_scale

    model_free_pars = [model_Rin, model_Rout, p]

    ### -------------------------------------------------------------------------------------------

    model_matrix = np.zeros((len(model_coords), len(model_coords)))

    for i, x in enumerate(model_coords):
        for j, y in enumerate(model_coords):
            radius = np.sqrt(x**2 + y**2) * (arcsec_per_pix * model_scale)
            if radius >= model_Rin and radius <= model_Rout:
                model_matrix[i, j] = thermalIntensitySingle(radius, model_sr_per_pix, fixed_pars, model_free_pars)

    print(model_matrix)

    model_centerPixel = [model_dimension / 2, model_dimension / 2]
    model_extent = [(-model_centerPixel[0]) * model_arcsec_per_pix, (model_dimension - model_centerPixel[0]) * model_arcsec_per_pix,
        (-model_centerPixel[1]) * model_arcsec_per_pix, (model_dimension - model_centerPixel[1]) * model_arcsec_per_pix]

    plt.figure("model")
    plt.imshow(model_matrix, origin = "lower", extent = model_extent, cmap = "inferno")
    plt.xlim([-1,1])
    plt.ylim([-1,1])

    plt.title(f"{model_scale}-scale model")
    plt.colorbar()

    ### -------------------------------------------------------------------------------------------

    total_intensity_radii = 100

    Rin = 0.01
    Rout = 30 / arcsec_per_pix

    radii = np.linspace(Rin, Rout, total_intensity_radii)
    arcsec_radii = np.linspace(Rin*arcsec_per_pix, Rout*arcsec_per_pix, total_intensity_radii)

    p_values = np.linspace(0, 3, 9)

    ### -------------------------------------------------------------------------------------------

    plt.figure("dust_surface_densities")
    for p in p_values:
        surface_density = []
        for radius in radii:
            surface_density.append(dustSurfaceDensitySingle(radius, Rin, Sig0, p))

        plt.plot(arcsec_radii, surface_density, label = f"p = {p:.2f}")

    plt.title("dust_surface_densities")
    plt.xlabel("Radius (arcseconds)")
    plt.yscale("log")
    plt.legend(loc = "best")

    ### -------------------------------------------------------------------------------------------

    plt.figure("dust_optical_depths")
    for p in p_values:
        optical_depth = []
        for radius in radii:
            optical_depth.append(dustOpticalDepthSingle(radius, Rin, Sig0, p, k, i))

        plt.plot(arcsec_radii, optical_depth, label = f"p = {p:.2f}")

    plt.title("dust_optical_depths")
    plt.xlabel("Radius (arcseconds)")
    plt.yscale("log")
    plt.legend(loc = "best")
    ### -------------------------------------------------------------------------------------------

    plt.figure("disk_temperatures")
    for p in p_values:
        disk_temperature = []
        for radius in radii:
            disk_temperature.append(diskTemperature(radius, Rin, T0, q))

        plt.plot(arcsec_radii, disk_temperature, label = f"p = {p:.2f}")

    plt.title("disk_temperatures")
    plt.xlabel("Radius (arcseconds)")
    plt.yscale("log")
    plt.legend(loc = "best")

    ### -------------------------------------------------------------------------------------------

    plt.figure("intensities")
    for p in p_values:
        free_pars = [Rin, Rout, p]
        intensity_profile = []
        for radius in radii:
            intensity_profile.append(thermalIntensitySingle(radius, sr_per_pix, fixed_pars, free_pars))

        plt.plot(arcsec_radii, intensity_profile, label = f"p = {p:.2f}")

    plt.title("intensities")
    plt.xlabel("Radius (arcseconds)")
    plt.yscale("log")
    plt.legend(loc = "best")

    ### -------------------------------------------------------------------------------------------

    plt.figure("intensities_normalised")
    for p in p_values:
        free_pars = [Rin, Rout, p]
        intensity_profile = []
        for radius in radii:
            intensity_profile.append(thermalIntensitySingle(radius, sr_per_pix, fixed_pars, free_pars))

        intensity_profile = np.asarray(intensity_profile)
        intensity_max = np.max(intensity_profile)

        plt.plot(arcsec_radii, intensity_profile / intensity_max, label = f"p = {p:.2f}")

    plt.title("intensities_normalised")
    plt.yscale("log")
    plt.legend(loc = "best")

    ### -------------------------------------------------------------------------------------------

    plt.show()
