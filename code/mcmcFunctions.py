# Michael Stroet  11293284

import numpy as np

from modelImage import *
from convolution import *
from astropy.convolution import convolve
from radialProfileCircle import getCircleProfile
from radialProfileEllipse import getEllipseProfile

def getDataIntensities(data, max_radius, eccentricity, rotation):

    convolved_image = convolveDataImage(data)

    data_intensities = np.asarray(getEllipseProfile(convolveDataImage(data), max_radius, eccentricity, rotation))
    data_max = np.max(data_intensities)

    # Return the intensities scaled in such a way that the peak is at 1
    return data_intensities / data_max

def getModelIntensities(free_pars, PIXEL_COORDS, radii, arcsec_per_pix, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL, model, crop_radius, crop = False):

    # generate the model image
    model_image = getImageMatrix(FIXED_PARS, free_pars, PIXEL_COORDS, arcsec_per_pix, SR_PER_PIX, model)
    if crop:
        model_image = cropImage(model_image, crop_radius)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, MODEL_KERNEL)

    # Generate a radial intensity profile from the model image
    model_intensities = np.asarray(getCircleProfile(convolved_model_image, radii))
    model_max = np.max(model_intensities)

    if model_max == np.nan or model_max <= 0:
        print(f"model_max: {model_max}")

    # Return the intensities scaled in such a way that the peak is at 1
    return model_intensities / model_max

# log-prior function
def logPrior(free_pars, arcsec_per_pix, parameter_ranges):

    for parameter, range in zip(free_pars, parameter_ranges(free_pars, arcsec_per_pix)):
        if not range[0] <= parameter <= range[1]:
            return -np.inf

    return 0

# log-likelihood function
def logLikelihood(free_pars, CONSTANTS):
    PIXEL_DIMENSION, PIXEL_COORDS, fit_radii, arcsec_per_pix, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL, model, crop_radius = CONSTANTS

    I_model = getModelIntensities(free_pars, PIXEL_COORDS, fit_radii, arcsec_per_pix, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL, model, crop_radius, True)
    variance = np.sum((FIT_DATA_INTENSITIES - I_model)**2) / len(FIT_DATA_INTENSITIES)

    log_likelihood = -0.5 * np.sum(((FIT_DATA_INTENSITIES - I_model)**2 / variance) + np.log(2 * np.pi * variance))

    return log_likelihood

# log-probability function
def logProbability(free_pars, CONSTANTS, parameter_ranges):
    PIXEL_DIMENSION, PIXEL_COORDS, fit_radii, arcsec_per_pix, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL, model, crop_radius = CONSTANTS

    log_prior = logPrior(free_pars, arcsec_per_pix, parameter_ranges)
    if np.isfinite(log_prior):
        log_likelihood = logLikelihood(free_pars, CONSTANTS)
        if np.isnan(log_likelihood):
            print(f"log_likelihood returned nan, return -infinity")
            return -np.inf

        return log_likelihood + log_prior
    else:
        return log_prior
