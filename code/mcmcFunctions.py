# Michael Stroet  11293284

import numpy as np

from modelImage import *
from convolution import *
from astropy.convolution import convolve
from radialProfileCircle import getCircleProfile
from radialProfileEllipse import getEllipseProfile

def getDataIntensities(data, radii, eccentricity, rotation):

    convolved_image = convolveDataImage(data)
    data_intensities = np.asarray(getEllipseProfile(convolved_image, radii, eccentricity, rotation))
    data_max = np.max(data_intensities)

    # Return the intensities scaled in such a way that the peak is at 1
    return data_intensities / data_max

def getModelIntensities(free_pars, pixel_coords, radii, arcsec_per_pix, sr_per_pix, fixed_pars, model_kernel, model, crop_radius, crop = False):

    # generate the model image
    model_image = getImageMatrix(fixed_pars, free_pars, pixel_coords, arcsec_per_pix, sr_per_pix, model)
    if crop:
        model_image = cropImage(model_image, crop_radius)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, model_kernel)

    # Generate a radial intensity profile from the model image
    model_intensities = np.asarray(getCircleProfile(convolved_model_image, radii))
    model_max = np.max(model_intensities)

    if model_max == np.nan or model_max <= 0:
        print(f"model_max: {model_max}")

    # Return the intensities scaled in such a way that the peak is at 1
    return model_intensities / model_max

# log-prior function
def logPrior(free_pars, pars_ranges):

    for parameter, range in zip(free_pars, pars_ranges):
        if not range[0] <= parameter <= range[1]:
            return -np.inf

    return 0

# log-likelihood function
def logLikelihood(free_pars, constants):
    pars_ranges, pixel_dimension, pixel_coords, fit_radii, arcsec_per_pix, sr_per_pix, FIT_DATA_INTENSITIES, fixed_pars, model_kernel, model, crop_radius = constants

    I_model = getModelIntensities(free_pars, pixel_coords, fit_radii, arcsec_per_pix, sr_per_pix, fixed_pars, model_kernel, model, crop_radius, True)
    variance = np.sum((FIT_DATA_INTENSITIES - I_model)**2) / len(FIT_DATA_INTENSITIES)

    log_likelihood = -0.5 * np.sum(((FIT_DATA_INTENSITIES - I_model)**2 / variance) + np.log(2 * np.pi * variance))

    return log_likelihood

# log-probability function
def logProbability(free_pars, constants):
    pars_ranges, pixel_dimension, pixel_coords, fit_radii, arcsec_per_pix, sr_per_pix, FIT_DATA_INTENSITIES, fixed_pars, model_kernel, model, crop_radius = constants

    log_prior = logPrior(free_pars, pars_ranges)
    if np.isfinite(log_prior):
        log_likelihood = logLikelihood(free_pars, constants)
        if np.isnan(log_likelihood):
            print(f"log_likelihood returned nan, return -infinity")
            return -np.inf

        return log_likelihood + log_prior
    else:
        return log_prior
