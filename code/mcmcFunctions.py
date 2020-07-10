# Michael Stroet  11293284

import numpy as np

from modelImage import *
from convolution import *
from astropy.convolution import convolve
from radialProfileCircle import getCircleProfile
from radialProfileEllipse import getEllipseProfile

def getDataIntensities(convolved_data, fit_radii, crop_radii, eccentricity, rotation, variance_range):

    data_intensities = np.asarray(getEllipseProfile(convolved_data, fit_radii, eccentricity, rotation))
    data_max = np.max(data_intensities)

    # Find indeces of the crop_radii in the variance range
    variance_intensities = []
    for i, radius in enumerate(crop_radii):
        if variance_range[0] <= radius <= variance_range[1]:
            variance_intensities.append(data_intensities[i])

    variance = sum(variance_intensities) / len(variance_intensities)

    # Return the intensities scaled in such a way that the peak is at 1 and
    # the variance scaled by the same factor
    return data_intensities / data_max, variance / data_max

def getModelIntensities(free_pars, fixed_pars, model_radii, model_coords, model_kernel, arcsec_per_pix, sr_per_pix, model):

    # generate the model image
    model_image = getImageMatrix(fixed_pars, free_pars, model_coords, arcsec_per_pix, sr_per_pix, model)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, model_kernel)

    # Generate a radial intensity profile from the model image
    model_intensities = np.asarray(getCircleProfile(convolved_model_image, model_radii))
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
    fixed_pars, pars_ranges, I_data, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, variance, model = constants

    I_model = getModelIntensities(free_pars, fixed_pars, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, model)
    log_likelihood = -0.5 * np.sum(((I_data - I_model)**2 / variance) + np.log(2 * np.pi * variance))

    return log_likelihood

# log-probability function
def logProbability(free_pars, constants):
    fixed_pars, pars_ranges, I_data, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, variance, model = constants

    log_prior = logPrior(free_pars, pars_ranges)
    if np.isfinite(log_prior):

        log_likelihood = logLikelihood(free_pars, constants)
        if np.isnan(log_likelihood):
            print(f"log_likelihood returned nan, return -infinity")
            return -np.inf

        return log_likelihood + log_prior

    else:
        return log_prior
