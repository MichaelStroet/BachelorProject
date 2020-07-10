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

def dustSurfaceDensityGaussian(R, Rin, Sig0, c):
    """
    Calculates the dust surface density (Sigma d) from a gaussian distribution.
    """
    return Sig0 * np.exp(-0.5 * pow((R - Rin) / c, 2))

### Disk temperature  -----------------------------------------------------------------------------------------------------------

def diskTemperature(R, Rin, T0, q):
    """
    Calculates the temperature at radius R from a single power law.
    """

    return T0 * pow(R / Rin, -q)

### Thermal continuum intensity -------------------------------------------------------------------------------------------------

def thermalIntensitySingle(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, q, Sig0 = fixed_pars
    Rin, Rout, p = free_pars

    T_disk = diskTemperature(R, Rin, T0, q)
    optical_depth = dustSurfaceDensitySingle(R, Rin, Sig0, p) * k * np.cos(i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensityDouble(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, q, Sig1, Rin, R1, p1 = fixed_pars
    Rout, SigFrac, p2 = free_pars
    Sig2 = SigFrac * Sig1

    T_disk = diskTemperature(R, Rin, T0, q)
    optical_depth = dustSurfaceDensityDouble(R, Sig1, Sig2, R1, p1, p2) * k * np.cos(i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))

def thermalIntensityGaussian(R, sr_per_pix, fixed_pars, free_pars):
    """
    Calculates the thermal intensity from the dust at radius R.
    """

    v, k, i, T0, q, Sig0 = fixed_pars
    Rin, Rout, c = free_pars

    T_disk = diskTemperature(R, Rin, T0, q)
    optical_depth = dustSurfaceDensityGaussian(R, Rin, Sig0, c) * k * np.cos(i)

    return planckFunction(v, T_disk, sr_per_pix) * (1 - np.exp(-optical_depth))
