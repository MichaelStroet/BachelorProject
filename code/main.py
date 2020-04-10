# Michael Stroet  11293284

import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add a path to the data folder
sys.path.append(os.path.join(root_directory, "data"))

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

def dustSurfaceDensitySingle(R):
    """
    Calculates the dust surface density (Sigma d) from a single power law.
    R is given in AU.
    """

    # Parameters
    Sigma0 = 3.9 # kg m^-2 (density at R0)
    R0 = 10 # AU
    p0 = 0.7

    return Sigma0 * pow(R / R0, -p0)

def dustSurfaceDensityBroken(R):
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

    return dustSurfaceDensityBroken(R) * k * np.cos(i)

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

def plotDiskIntensity():
    pass

if __name__ == "__main__":

    # Parameters
    frequency = 365.5e9 # Hz
    inclination = 0.0*np.pi # [0, np.pi/2]
    Rinner = 1 # AU
    Router = 200 # AU

    coords = np.linspace(-Router, Router, 1000)
    values = np.zeros((len(coords), len(coords)))

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            radius = np.sqrt(x**2 + y**2)
            if radius >= Rinner and radius <= Router:
                values[i, j] = thermalIntensity(frequency, radius, inclination)

    plt.imshow(values, cmap="inferno", extent = [coords[-1], coords[1], coords[1], coords[-1]])

    plt.title("An attempt was made")
    plt.xlabel("X [AU]")
    plt.ylabel("Y [AU]")

    cbar = plt.colorbar()
    cbar.set_label("test")

    plt.show()
