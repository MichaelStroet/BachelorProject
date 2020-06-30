# Michael Stroet  11293284

# Extracts a radial profile from a 2D image using concentric ellipses

import numpy as np
from scipy.ndimage import map_coordinates

def getUnits(pixelSize):
    if pixelSize == 1:
        return "pixels"
    else:
        return "also pixels"

def getEllipseCoordinates(center, major, ecc, theta):
    """
    Returns coordinates for a ellipse with eccentricity 'ecc'
    and semi-major axis 'major' centered at 'center' and rotated by angle 'theta'
    """

    angles = np.linspace(0, 2*np.pi, 1000)
    minor = major * np.sqrt(1 - ecc**2)

    coordinates = [[], []]
    for angle in angles:
        coordinates[0].append(center[0] + (major * np.cos(angle) * np.cos(theta)) - (minor * np.sin(angle) * np.sin(theta)))
        coordinates[1].append(center[1] + (major * np.cos(angle) * np.sin(theta)) + (minor * np.sin(angle) * np.cos(theta)))

    return coordinates

def getEllipseAverage(image, semi_major, eccentricity, rotation):
    """
    """

    center = [(image.shape[0] - 1) / 2., (image.shape[1] - 1) / 2.]
    coordinates = getEllipseCoordinates(center, semi_major, eccentricity, rotation)

    # Extract the values along the ellipse, using cubic interpolation
    values = map_coordinates(image, coordinates)
    average = sum(values) / len(values)

    return abs(average)

def getEllipseProfile(image, pixel_radii, eccentricity, rotation):
    """
    Creates a radial profile of an image by averaging values of ellipses.
    """

    radial_profile = []
    for semi_major in pixel_radii:
        radial_profile.append(getEllipseAverage(image, semi_major, eccentricity, rotation))

    return radial_profile
