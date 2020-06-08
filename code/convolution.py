# Michael Stroet  11293284

import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel

def convolveImageGaussian2D(data, theta):
    """
    Convolve the image with a 2D Gaussian kernel.
    image is a nd numpy array,
    major and minor are the lengths of the axes of the FWHM ellipse in pixels,
    theta is the rotation angle in radians.
    """
    image, header, wcs = data

    major_axis = (header["beamSemiMajor"] * 2) / header["degreesPixelScale"] # pixels
    minor_axis = (header["beamSemiMinor"] * 2) / header["degreesPixelScale"] # pixels

    FWHM_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_x = major_axis / FWHM_factor
    sigma_y = minor_axis / FWHM_factor

    return convolve(image, Gaussian2DKernel(sigma_x, sigma_y, theta))

def convolveDataImage(data):
    image, header, wcs = data

    major_axis = (header["beamSemiMajor"] * 2) / header["degreesPixelScale"] # pixels
    minor_axis = (header["beamSemiMinor"] * 2) / header["degreesPixelScale"] # pixels

    FWHM_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_x = major_axis / FWHM_factor
    sigma_y = minor_axis / FWHM_factor

    angle = header['beamPA'] * (np.pi / 180) # radian (PA already defined as 90 degrees)
    kernel = Gaussian2DKernel(sigma_x, sigma_y, angle)

    return convolve(image, kernel)

def generateModelKernels(data):
    image, header, wcs = data

    major_axis = (header["beamSemiMajor"] * 2) / header["degreesPixelScale"] # pixels
    minor_axis = (header["beamSemiMinor"] * 2) / header["degreesPixelScale"] # pixels

    FWHM_factor = 2 * np.sqrt(2 * np.log(2))
    sigma_x = major_axis / FWHM_factor
    sigma_y = minor_axis / FWHM_factor

    perp_angle = (header['beamPA'] - 90) * (np.pi / 180) # radian
    perp_kernel = Gaussian2DKernel(sigma_x, sigma_y, perp_angle)

    beam_angle = perp_angle - (0.5 * np.pi)
    beam_kernel = Gaussian2DKernel(sigma_x, sigma_y, beam_angle)

    area_kernel = convolve(beam_kernel, perp_kernel)
    peak_kernel = convolve(beam_kernel, perp_kernel)
    area_kernel.normalize(mode = 'integral')
    peak_kernel.normalize(mode = 'peak')

    return area_kernel, peak_kernel
