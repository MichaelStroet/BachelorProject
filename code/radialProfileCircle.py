# Michael Stroet  11293284

# Extracts a radial profile from a 2D image using concentric circles

import numpy as np
from scipy.ndimage import map_coordinates

def getCircleCoordinates(center, R):
    """
    Returns coordinates for a circle of radius 'R' centered at 'center'
    """

    angles = np.linspace(0, 2*np.pi, 1000)

    coordinates = [[], []]
    for angle in angles:
        coordinates[0].append(center + R * np.cos(angle))
        coordinates[1].append(center + R * np.sin(angle))

    return coordinates

def getCircleAverage(image, R):
    """
    Determines the average value of points on a circle of radius 'R' centered at 'center'
    'R' and 'center' are given in pixel values.
    """

    center = (image.shape[0] - 1) / 2.
    x_coords, y_coords = getCircleCoordinates(center, R)

    # Extract the values along the circle, using cubic interpolation
    values = map_coordinates(image, [x_coords, y_coords])
    average = sum(values) / len(values)

    return abs(average)

def getCircleProfile(image, pixel_radii):
    """
    Creates a radial profile of an image by averaging values of circles.
    """

    radial_profile = []
    for radius in pixel_radii:
        radial_profile.append(getCircleAverage(image, radius))

    return radial_profile
