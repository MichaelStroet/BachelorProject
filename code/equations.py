# Michael Stroet  11293284

# This script contains the equations used in the model
# Run the script itself to generate a plot for each equation

import numpy as np

def planckFunction(v, T, sr_per_pix):
    """
    Returns the intensity of thermal radiation a black body emits
    at frequency v and temperature T.
    """

    # Constants
    h = 6.626e-34 # J s (Planck"s constant)
    c = 2.998e8 # m s^-1 (Speed of light)
    kB = 1.38e-23 # J K^-1 (Boltzmann"s constant)

    # Calculate intensity in J/m^2/sr
    planck = 2 * h * v**3 / c**2 * pow(np.exp(h * v / kB / T) - 1, -1)

    # Convert J/m^2/sr to Jy/pixel
    return 1e26 * sr_per_pix * planck

def dustSurfaceDensity(R, Sig0, R_br, p0, p1):
    """
    Calculates the dust surface density (Sigma d) from a broken power law.
    """

    if R <= R_br:
        return Sig0 * pow(R / R_br, -p0)
    else:
        return Sig0 * pow(R / R_br, -p1)

def dustOpticalDepth(R, Sig0, R_br, p0, p1, k, i):
    """
    Calculates the dust optical depth (tau) for radius R.
    """
    return dustSurfaceDensity(R, Sig0, R_br, p0, p1) * k * np.cos(i)

def diskTemperature(R, R0, T0, q0, q1):
    """
    Calculates the temperature at radius R from a broken power law.
    """

    if R <= R0:
        return T0 * pow(R / R0, -q0)
    else:
        return T0 * pow(R / R0, -q1)

def thermalIntensity(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """
    v, k, i = fixed_pars
    Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0 = free_pars

    T_disk = diskTemperature(R, R0, T0, q0, q1)
    optical_depth = dustOpticalDepth(R, Sig0, R_br, p0, p1, k, i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

if __name__ == "__main__":

    import os, sys
    import matplotlib.pyplot as plt

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    # Parameters
    v = 365.5e9 # Hz
    R0 = 7 # AU
    T0 = 27 # K
    q0 = 2.6
    q1 = 0.26
    k = 0.34 # m^2 kg^-1 (at 365.5 GHz)
    Sig0 = 0.1 # kg m^-2 (guess)
    R_br = 47 # AU
    p0 = 0.53
    p1 = 8.0
    i = 0.0*np.pi # [0, np.pi/2]

    R = 10 # AU
    Rin = 2 # AU
    Rout = 200 # AU

    sr_per_pix = 9.4e-13
    fixed_pars = (v, k, i)
    free_pars = [Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0]

    # Test planckFunction

    temperature = diskTemperature(R, R0, T0, q0, q1)

    frequencies = np.linspace(0.1, 1e13, 1000)
    intensities = []

    for frequency in frequencies:
        intensities.append(planckFunction(frequency, temperature, sr_per_pix))

    plt.figure("planckFunction", figsize = (10, 5))

    plt.plot(frequencies / 1e9, intensities)

    plt.title(f"planckFunction, R = {R} AU -> T = {temperature:.2f} K")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Intensity [Jy/pixel]")

    plt.savefig(figures_directory + pyName() + "-" + "planckFunction.png")

    # Test dustSurfaceDensity

    radii = np.linspace(Rin, Rout, 1000)
    densities = []

    for radius in radii:
        densities.append(dustSurfaceDensity(radius, Sig0, R_br, p0, p1))

    plt.figure("dustSurfaceDensity", figsize = (10, 5))

    plt.plot(radii, densities)
    plt.yscale('log')

    plt.title("dustSurfaceDensity")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Surface density [kg/m^2]")

    plt.savefig(figures_directory + pyName() + "-" + "dustSurfaceDensity.png")

    # Test dustOpticalDepth

    radii = np.linspace(Rin, Rout, 1000)
    optical_depths = []

    for radius in radii:
        optical_depths.append(dustOpticalDepth(radius, Sig0, R_br, p0, p1, k, i))

    plt.figure("dustOpticalDepth", figsize = (10, 5))

    plt.plot(radii, optical_depths)
    plt.yscale('log')

    plt.title(f"dustOpticalDepth, i = {i/np.pi} π")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Optical depth")

    plt.savefig(figures_directory + pyName() + "-" + "dustOpticalDepth.png")


    # Test diskTemperature

    radii = np.linspace(Rin, Rout, 1000)
    temperatures = []

    for R in radii:
        temperatures.append(diskTemperature(R, R0, T0, q0, q1))

    plt.figure("diskTemperature", figsize = (10, 5))

    plt.plot(radii, temperatures)
    plt.yscale('log')

    plt.title("diskTemperature")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Temperature [K]")

    plt.savefig(figures_directory + pyName() + "-" + "diskTemperature.png")


    # Test thermalIntensity

    radii = np.linspace(Rin, Rout, 1000)
    thermal_intensities = []

    for radius in radii:
        thermal_intensities.append(thermalIntensity(radius, sr_per_pix, fixed_pars, free_pars))

    plt.figure("thermalIntensity", figsize = (10, 5))

    plt.plot(radii, thermal_intensities)
    plt.yscale('log')

    plt.title(f"Thermal continuum, v = {v/1e9}GHz, i = {i/np.pi}π")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Intensity [Jy/pixel]")

    plt.savefig(figures_directory + pyName() + "-" + "thermalIntensity.png")

    plt.show()
