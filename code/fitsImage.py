# Michael Stroet  11293284

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates

def plotFitsImage(image):
    """
    Plot a 2D image of a nd numpy array
    """

    plt.imshow(image, cmap="inferno")
    rows, cols = np.where(np.isnan(image))
    plt.plot(cols, rows, 'gx', markersize=10)

    plt.title("Thermal Continuum disk")
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")

    cbar = plt.colorbar()
    cbar.set_label("Intensity")

def getLineCoordinates(pixelDimension, radius, angle):
    """
    """

    center = (pixelDimension - 1) / 2.0
    edge = [center + radius * np.cos(angle), center + radius * np.sin(angle)]

    return center, edge

def getFitsRadialProfile(image, pixelDimension, profileRadius, angle):
    """
    """

    center, edge = getLineCoordinates(pixelDimension, profileRadius, angle + (np.pi / 2))
    N_points = pixelDimension

    # Generate N coordinates between the center and edge points
    x, y = np.linspace(center, edge[0], N_points), np.linspace(center, edge[1], N_points)

    print(np.vstack((x,y)))

    # Extract the values along the line, using cubic interpolation
    image_profile = map_coordinates(image, np.vstack((x,y)), mode = "nearest")

    return image_profile

def plotFitsRadialProfileWithImage(image, pixelDimension, profileRadius, angle):
    """
    """

    print("\nGenerating image profile:\n")
    image_profile = getFitsRadialProfile(image, pixelDimension, profileRadius, angle)

    with np.printoptions(threshold=np.inf):
        print(image_profile)

    center, edge = getLineCoordinates(pixelDimension, profileRadius, angle)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 7]}, figsize = (15, 6))

    ax1.imshow(image, cmap="inferno")
    ax1.plot([center, edge[0]], [center, edge[1]], "go-")
    ax1.set_title("Image")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")

    ax2.plot(image_profile)
    ax2.set_title("Intensity radial profile")
    ax2.set_xlabel("R [pixels]")
    ax2.set_ylabel("Intensity")
