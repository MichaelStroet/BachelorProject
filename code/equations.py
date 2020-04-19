# Michael Stroet  11293284

# This script contains the equations used in the model
# Run the script itself to generate a plot for each equation

import numpy as np
import matplotlib.pyplot as plt

def planckFunction(v, R):
    """
    Returns the intensity of thermal radiation a black body emits
    at frequency nu (v) and temperature T(R).
    v is given in Hz, R in AU.
    """

    # Constants
    h = 6.626e-34 # m^2 kg s^-1 (Planck"s constant)
    c = 2.998e8 # m s^-1 (Speed of light)
    kB = 1.38e-23 # J K^-1 (Boltzmann"s constant)

    return 2 * h * pow(v, 3) / pow(c, 2) * pow(np.exp(h * v / kB / diskTemperature(R)) - 1, -1)

def dustSurfaceDensity(R):
    """
    Calculates the dust surface density (Sigma d) from a broken power law.
    R is given in AU.
    """

    # Parameters
    Sigma0 = 0.1 # kg m^-2 (guess)
    Rbreak = 47 # AU
    p0 = 0.53
    p1 = 8.0


    if R <= Rbreak:
        return Sigma0 * pow(R / Rbreak, -p0)
    else:
        return Sigma0 * pow(R / Rbreak, -p1)

def dustOpticalDepth(R, i):
    """
    Calculates the dust optical depth (tau) for radius R at 365.5 GHz.
    R is given in AU, i in radians [0, π/2].
    """

    # Parameters
    k = 0.34 # m^2 kg^-1 (at 365.5 GHz)

    return dustSurfaceDensity(R) * k * np.cos(i)

def diskTemperature(R):
    """
    Calculates the temperature at radius R from a broken power law.
    R is given in AU.
    """

    # Parameters
    R0 = 7 # AU
    T0 = 27 # K
    q0 = 2.6
    q1 = 0.26

    if R <= R0:
        return T0 * pow(R / R0, -q0)
    else:
        return T0 * pow(R / R0, -q1)


def thermalIntensity(v, R, i):
    """
    Calculates the thermal intensity at frequency v from the dust
    inside the disk at radius R
    v is given in Hz, R in AU, i in radians [0, π/2]
    """

    return planckFunction(v, diskTemperature(R)) * (1 - np.exp(-1 * dustOpticalDepth(R, i)))

if __name__ == "__main__":

    import os, sys

    def pyName():
        return __file__.split("\\")[-1].replace(".py", "")

    root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    figures_directory = root_directory + "\\data\\codeFigures\\"

    # Test planckFunction

    radius = 10 # AU
    temperature = diskTemperature(radius)

    frequencies = np.linspace(0.1, 1e13, 1000)
    intensities = []

    for frequency in frequencies:
        intensities.append(planckFunction(frequency, temperature))

    plt.figure("planckFunction", figsize = (10, 5))

    plt.plot(frequencies / 1e9, intensities)

    plt.title(f"planckFunction, R = {radius} AU -> T = {temperature:.2f} K")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Intensity")

    plt.savefig(figures_directory + pyName() + "-" + "planckFunction.png")

    # Test dustSurfaceDensity

    Rinner = 2 # AU
    Router = 200 # AU

    radii = np.linspace(Rinner, Router, 1000)
    densities = []

    for radius in radii:
        densities.append(dustSurfaceDensity(radius))

    plt.figure("dustSurfaceDensity", figsize = (10, 5))

    plt.plot(radii, densities)

    plt.title("dustSurfaceDensity")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Surface density [kg/m^2]")

    plt.savefig(figures_directory + pyName() + "-" + "dustSurfaceDensity.png")

    # Test dustOpticalDepth

    inclination = 0 # [0, np.pi/2]
    Rinner = 2 # AU
    Router = 200 # AU

    radii = np.linspace(Rinner, Router, 1000)
    optical_depths = []

    for radius in radii:
        optical_depths.append(dustOpticalDepth(radius, inclination))

    plt.figure("dustOpticalDepth", figsize = (10, 5))

    plt.plot(radii, optical_depths)

    plt.title(f"dustOpticalDepth, i = {inclination/np.pi} π")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Optical depth")

    plt.savefig(figures_directory + pyName() + "-" + "dustOpticalDepth.png")


    # Test diskTemperature

    Rinner = 2 # AU
    Router = 200 # AU

    radii = np.linspace(Rinner, Router, 1000)
    temperatures = []

    for radius in radii:
        temperatures.append(diskTemperature(radius))

    plt.figure("diskTemperature", figsize = (10, 5))

    plt.plot(radii, temperatures)

    plt.title("diskTemperature")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Temperature [K]")

    plt.savefig(figures_directory + pyName() + "-" + "diskTemperature.png")


    # Test thermalIntensity

    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    Rinner = 1 # AU
    Router = 200 # AU

    radii = np.linspace(Rinner, Router, 1000)
    thermal_intensities = []

    for radius in radii:
        thermal_intensities.append(thermalIntensity(frequency, radius, inclination))

    plt.figure("thermalIntensity", figsize = (10, 5))

    plt.plot(radii, thermal_intensities)

    plt.title(f"Thermal continuum, v = {frequency/1e9}GHz, i = {inclination/np.pi}π")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Intensity")

    plt.savefig(figures_directory + pyName() + "-" + "thermalIntensity.png")

    plt.show()
