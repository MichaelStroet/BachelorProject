# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add a path to the data folder
sys.path.append(os.path.join(root_directory, "data"))

from main import planckFunction, dustSurfaceDensity, dustOpticalDepth, diskTemperature, thermalIntensity

if __name__ == "__main__":

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


    # Test thermalIntensity

    frequency = 365.5e9 # Hz
    inclination = 0.3*np.pi # [0, np.pi/2]
    Rinner = 2 # AU
    Router = 200 # AU

    radii = np.linspace(Rinner, Router, 1000)
    thermal_intensities = []

    for radius in radii:
        thermal_intensities.append(thermalIntensity(frequency, radius, inclination))

    plt.figure("thermalIntensity", figsize = (10, 5))

    plt.plot(radii, thermal_intensities)

    plt.title(f"thermalIntensity, v = {frequency/1e9}GHz, i = {inclination/np.pi}π")
    plt.xlabel("Radius [AU]")
    plt.ylabel("Intensity")

    plt.show()
