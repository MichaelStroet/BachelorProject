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

def dustSurfaceDensityTWHya(R, Sig0, R_br, p0, p1):
    """
    Calculates the dust surface density (Sigma d) from a 2-sloped power law.
    """

    if R <= R_br:
        return Sig0 * pow(R / R_br, -p0)
    else:
        return Sig0 * pow(R / R_br, -p1)

def dustSurfaceDensitySingle(R, Rin, Sig0, p):
    """
    Calculates the dust surface density (Sigma d) from single power law.
    """

    return Sig0 * pow(R / Rin, -p)

def dustSurfaceDensityDouble(R, Sig1, SigFrac, R1, p1, p2):
    """
    Calculates the dust surface density (Sigma d) from a 2-sloped discontinuous power law.
    """


    if R <= R1:
        return Sig1 * pow(R / R1, -p1)
    else:
        return SigFrac * Sig1 * pow(R / R1, -p2)

### Dust optical depth ----------------------------------------------------------------------------------------------------------

def dustOpticalDepthTWHya(R, Sig0, R_br, p0, p1, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensityTWHya(R, Sig0, R_br, p0, p1) * k * np.cos(i)

def dustOpticalDepthSingle(R, Rin, Sig0, p, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensitySingle(R, Rin, Sig0, p) * k * np.cos(i)

def dustOpticalDepthDouble(R, Sig1, SigFrac, R1, p1, p2, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensityDouble(R, Sig1, SigFrac, R1, p1, p2) * k * np.cos(i)

### Disk temperature  -----------------------------------------------------------------------------------------------------------

def diskTemperatureTWHya(R, R0, T0, q0, q1):
    """
    Calculates the temperature at radius R from a broken power law.
    """

    if R <= R0:
        return T0 * pow(R / R0, -q0)
    else:
        return T0 * pow(R / R0, -q1)

def diskTemperatureSingle(R, Rin, T0, q):
    """
    Calculates the temperature at radius R from a single power law.
    """

    return T0 * pow(R / Rin, -q)

### Thermal continuum intensity -------------------------------------------------------------------------------------------------

def thermalIntensityTWHya(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i = fixed_pars
    Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0 = free_pars

    T_disk = diskTemperatureTWHya(R, R0, T0, q0, q1)
    optical_depth = dustOpticalDepthTWHya(R, Sig0, R_br, p0, p1, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensitySingle(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, Sig0 = fixed_pars
    Rin, Rout, p, q = free_pars

    T_disk = diskTemperatureSingle(R, Rin, T0, q)
    optical_depth = dustOpticalDepthSingle(R, Rin, Sig0, p, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensityDouble(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, Sig1 = fixed_pars
    Rin, Rout, SigFrac, R1, p1, p2, q = free_pars

    T_disk = diskTemperatureSingle(R, Rin, T0, q)
    optical_depth = dustOpticalDepthDouble(R, Sig1, SigFrac, R1, p1, p2, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

# if __name__ == "__main__":
#
#     import os, sys
#     import matplotlib.pyplot as plt
#
#     def pyName():
#         return __file__.split("\\")[-1].replace(".py", "")
#
#     root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#     figures_directory = root_directory + "\\data\\codeFigures\\"
#
#     # Parameters
#     v = 365.5e9 # Hz
#     R0 = 7 # AU
#     T0 = 27 # K
#     q = 0.5
#     q0 = 2.6
#     q1 = 0.26
#     k = 0.34 # m^2 kg^-1 (at 365.5 GHz)
#     Sig0 = 0.25 # kg m^-2 (guess)
#     Sig1 = 1 # kg m^-2 (guess)
#     Sig2 = 0.3 # kg m^-2 (guess)
#     R_br = 47 # AU
#     R_1 = R_br
#     p = 2
#     p0 = 0.53
#     p1 = 8.0
#     p2 = 5
#     i = 0.0*np.pi # [0, np.pi/2]
#
#     R = 10 # AU
#     Rin = 2 # AU
#     Rout = 200 # AU
#
#     sr_per_pix = 9.4e-13
#     fixed_pars = (v, k, i)
#     free_pars = [Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0]
#
#     # Test planckFunction
#     temperature = diskTemperatureSingle(R, Rin, T0, q)
#
#     frequencies = np.linspace(0.1, 1e13, 1000)
#     intensities = []
#
#     for frequency in frequencies:
#         intensities.append(planckFunction(frequency, temperature, sr_per_pix))
#
#     plt.figure("planckFunction", figsize = (10, 5))
#
#     plt.plot(frequencies / 1e9, intensities)
#
#     plt.title(f"planckFunction, R = {R} AU -> T = {temperature:.2f} K")
#     plt.xlabel("Frequency [GHz]")
#     plt.ylabel("Intensity [Jy/pixel]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "planckFunction.png")
#
#     # Test dustSurfaceDensityTWHya
#
#     radii = np.linspace(Rin, Rout, 1000)
#     densities = []
#
#     for radius in radii:
#         densities.append(dustSurfaceDensityTWHya(radius, Sig0, R_br, p0, p1))
#
#     plt.figure("dustSurfaceDensityTWHya", figsize = (10, 5))
#
#     plt.plot(radii, densities)
#     plt.yscale('log')
#
#     plt.title("dustSurfaceDensityTWHya")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Surface density [kg/m^2]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "dustSurfaceDensityTWHya.png")
#
#     # Test dustSurfaceDensitySingle
#
#     radii = np.linspace(Rin, Rout, 1000)
#     densities = []
#
#     for radius in radii:
#         densities.append(dustSurfaceDensitySingle(radius, Sig0, R_br, p))
#
#     plt.figure("dustSurfaceDensitySingle", figsize = (10, 5))
#
#     plt.plot(radii, densities)
#     plt.yscale('log')
#
#     plt.title("dustSurfaceDensitySingle")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Surface density [kg/m^2]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "dustSurfaceDensitySingle.png")
#
#     # Test dustSurfaceDensityDouble
#
#     radii = np.linspace(Rin, Rout, 1000)
#     densities = []
#
#     for radius in radii:
#         densities.append(dustSurfaceDensityDouble(radius, Sig1, Sig2, R_1, p1, p2))
#
#     plt.figure("dustSurfaceDensityDouble", figsize = (10, 5))
#
#     plt.plot(radii, densities)
#     plt.yscale('log')
#
#     plt.title("dustSurfaceDensityDouble")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Surface density [kg/m^2]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "dustSurfaceDensityDouble.png")
#
#     # Test dustOpticalDepth
#
#     radii = np.linspace(Rin, Rout, 1000)
#     optical_depths = []
#
#     for radius in radii:
#         optical_depths.append(dustOpticalDepth(radius, Sig0, R_br, p0, p1, k, i))
#
#     plt.figure("dustOpticalDepth", figsize = (10, 5))
#
#     plt.plot(radii, optical_depths)
#     plt.yscale('log')
#
#     plt.title(f"dustOpticalDepth, i = {i/np.pi} π")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Optical depth")
#
#     plt.savefig(figures_directory + pyName() + "-" + "dustOpticalDepth.png")
#
#
#     # Test diskTemperatureSingle
#     radii = np.linspace(Rin, Rout, 1000)
#     temperatures = []
#
#     for R in radii:
#         temperatures.append(diskTemperatureSingle(R, Rin, T0, q))
#
#     plt.figure("diskTemperatureSingle", figsize = (10, 5))
#
#     plt.plot(radii, temperatures)
#     plt.yscale('log')
#
#     plt.title("diskTemperatureSingle")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Temperature [K]")
#
#     # Test diskTemperatureDouble
#     radii = np.linspace(Rin, Rout, 1000)
#     temperatures = []
#
#     for R in radii:
#         temperatures.append(diskTemperatureDouble(R, R_1, T0, q0, q1))
#
#     plt.figure("diskTemperatureDouble", figsize = (10, 5))
#
#     plt.plot(radii, temperatures)
#     plt.yscale('log')
#
#     plt.title("diskTemperatureDouble")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Temperature [K]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "diskTemperatureDouble.png")
#
#     # Test thermalIntensity
#
#     radii = np.linspace(Rin, Rout, 1000)
#     thermal_intensities = []
#
#     for radius in radii:
#         thermal_intensities.append(thermalIntensity(radius, sr_per_pix, fixed_pars, free_pars))
#
#     plt.figure("thermalIntensity", figsize = (10, 5))
#
#     plt.plot(radii, thermal_intensities)
#     plt.yscale('log')
#
#     plt.title(f"Thermal continuum, v = {v/1e9}GHz, i = {i/np.pi}π")
#     plt.xlabel("Radius [AU]")
#     plt.ylabel("Intensity [Jy/pixel]")
#
#     plt.savefig(figures_directory + pyName() + "-" + "thermalIntensity.png")
#
#     plt.show()
