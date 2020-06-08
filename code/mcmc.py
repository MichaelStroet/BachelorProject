# Michael Stroet  11293284

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

from fitsFiles import *
from modelImage import *
from convolution import *
from matplotlib.colors import LogNorm
from astropy.convolution import convolve
from radialProfileCircle import getCircleProfile

def getDataIntensities(data, radii):

    convolved_image = convolveDataImage(data)
    return np.asarray(getCircleProfile(convolveDataImage(data), radii))

def getModelIntensities(fixed_pars, free_pars):

    # generate the model image
    model_image = getImageMatrix(fixed_pars, free_pars, PIXEL_COORDS, SR_PER_PIX)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, MODEL_KERNEL)

    # Generate a radial intensity profile from the model image
    return np.asarray(getCircleProfile(convolved_model_image, PIXEL_RADII))

def parameter_ranges(free_pars):
    Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0 = free_pars

    ranges = []
    ranges.append([0.1 / ARCSEC_PER_PIX, Rout]) # Rin (pixels)
    ranges.append([Rin, 30 / ARCSEC_PER_PIX])   # Rout (pixels)
    ranges.append([10,50])                   # T0 (K)
    ranges.append([Rin,Rout])                # R_br (pixels)
    ranges.append([0,1])                    # p0
    ranges.append([3,15])                    # p1
    ranges.append([0.01,1])                  # Sig0 (kg m^-2)
    ranges.append([0,10])                    # q0
    ranges.append([0,10])                    # q1
    ranges.append([Rin, Rout])               # R0 (pixels)

    return ranges

# log-prior function
def logPrior(free_pars):
    for parameter, range in zip(free_pars, parameter_ranges(free_pars)):
        if not range[0] <= parameter <= range[1]:
            return -np.inf

    return 0

# log-likelihood function
def logLikelihood(free_pars, fixed_pars, I_data):
    I_model = getModelIntensities(fixed_pars, free_pars)
    variance = np.sum((I_data - I_model)**2) / len(I_data)

    return -0.5 * np.sum(((I_data - I_model)**2 / variance) + np.log(2 * np.pi * variance))

# log-probability function
def logProbability(free_pars, fixed_pars, I_data):
    log_prior = logPrior(free_pars)
    if np.isfinite(log_prior):
        log_likelihood = logLikelihood(free_pars, fixed_pars, I_data)
        return log_likelihood + log_prior
    else:
        return log_prior

def mcmc(data):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo")
    print("\n----------------------------------------------------------------------------------------------------\n")

    ### Generate global constants -----------------------------------------------------------------------------------------------

    global PIXEL_DIMENSION
    global PIXEL_RADIUS
    global PIXEL_COORDS
    global PIXEL_RADII
    global ARCSEC_PER_PIX
    global SR_PER_PIX

    total_intensity_radii = 100

    PIXEL_DIMENSION = min(data[0].shape)
    PIXEL_RADIUS = PIXEL_DIMENSION / 2
    PIXEL_COORDS = np.linspace(-PIXEL_RADIUS, PIXEL_RADIUS, PIXEL_DIMENSION)
    PIXEL_RADII = np.linspace(0, PIXEL_RADIUS, total_intensity_radii)
    ARCSEC_PER_PIX = data[1]["degreesPixelScale"] * 3600
    SR_PER_PIX = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    data_intensities = getDataIntensities(data, PIXEL_RADII)

    ### Setup model -------------------------------------------------------------------------------------------------------------

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = 42.46 * (np.pi / 180) # radian

    # Free parameters guesses
    Rin = 2.3 # Pixels
    Rout = 125 # Pixels
    T0 = 27 # K
    R_br = 75 # Pixels
    p0 = 0.53
    p1 = 8.0
    Sig0 = 0.1 # kg m^-2 (random guess)
    q0 = 2.6
    q1 = 0.3
    R0 = 15 # Pixels --> radius units dont matter, only their fraction

    fixed_pars = (v, k, i)
    free_pars = [Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0]
    free_labels = ["Rin", "Rout", "T0", "R_br", "p0", "p1", "Sig0", "q0", "q1", "R0"]

    # Generate the convolution kernels for the model image
    global MODEL_KERNEL
    model_kernel_area, model_kernel_peak = generateModelKernels(data)

    MODEL_KERNEL = model_kernel_area
    kernel_used = "area"

    print(f"\nModel kernel used is: {kernel_used}")

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    nwalkers = 100
    ndim = len(free_pars)
    args = (fixed_pars, data_intensities)

    print(f"\nInitialize sampler with {nwalkers} walkers and {ndim} free parameters ({free_labels})")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=args)

    ### Burn-in sampler ---------------------------------------------------------------------------------------------------------

    start_locations = free_pars + 1e-2 * np.random.randn(nwalkers, ndim)
    burnin_steps = 300

    print(f"\nRunning {burnin_steps} burn-in steps:")
    burnin_state = sampler.run_mcmc(start_locations, burnin_steps, progress = True)
    burnin_samples = sampler.get_chain(flat=False)
    sampler.reset()

    ### Production run ----------------------------------------------------------------------------------------------------------

    production_steps = 600

    print(f"\nRunning {production_steps} production steps:")
    sampler.run_mcmc(burnin_state, production_steps, progress = True)

    flat_samples = sampler.get_chain(flat=True)

    # Get the 50 percentile parameter values
    mcmc_pars = []
    for i in range(ndim):
        mcmc_pars.append(np.percentile(flat_samples[:, i], 50))

    ### Visualise results -------------------------------------------------------------------------------------------------------

    # Visualise burnin steps for each parameter
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True, num = f"Burnin - {kernel_used}")
    for i, label in enumerate(free_labels):
        ax = axes[i]
        ax.plot(burnin_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(burnin_samples))
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    # original and convolved model image from mcmc parameters
    model_intensities = getModelIntensities(fixed_pars, mcmc_pars)

    data_image = data[0]
    data_image[np.where(data_image <= 0.0)] = np.min(data_image[np.where(data_image > 0)])

    convolved_data = convolveDataImage(data)
    convolved_data[np.where(convolved_data <= 0.0)] = np.min(convolved_data[np.where(convolved_data > 0)])

    model_image = getImageMatrix(fixed_pars, mcmc_pars, PIXEL_COORDS, SR_PER_PIX)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])

    convolved_model = convolve(model_image, MODEL_KERNEL)
    convolved_model[np.where(convolved_model <= 0.0)] = np.min(convolved_model[np.where(convolved_model > 0)])

    centerPixel = (data[1]["xCenterPixel"], data[1]["yCenterPixel"])
    pixelDimension = data[0].shape

    extent = [(-centerPixel[0]) * ARCSEC_PER_PIX, (pixelDimension[0] - centerPixel[0]) * ARCSEC_PER_PIX,
        (-centerPixel[1]) * ARCSEC_PER_PIX, (pixelDimension[1] - centerPixel[1]) * ARCSEC_PER_PIX]

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = "Images")

    ax1.imshow(data_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax1.set_title("Original data")

    ax2.imshow(model_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax2.set_title("Original model")

    ax3.imshow(convolved_data, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax3.set_title("Convolved data")

    ax4.imshow(convolved_model, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax4.set_title("Convolved model")

    # Histogram plots of parameters
    fig, axes =  plt.subplots(1, ndim, num = f"ParameterHistograms - {kernel_used}")
    fig.suptitle(f"Parameter probabilities after {production_steps} steps")

    for i, ax in enumerate(axes):
        y_hist, x_hist, _ = ax.hist(flat_samples[:, i], 100, color="k", histtype="step")
        ax.set_title(free_labels[i])
        ax.set_xlabel(free_labels[i])
        ax.set_yticks([])

    # Corner plot
    fig = corner.corner(flat_samples, labels=free_labels, truths=free_pars, show_titles=True, quantiles=[0.16, 0.50, 0.84])
    fig.canvas.set_window_title(f"CornerPlot - {kernel_used}")

    # Intensity profile comparison
    arcsec_radius = PIXEL_RADIUS * ARCSEC_PER_PIX
    arcsec_radii = np.linspace(0, arcsec_radius, total_intensity_radii)

    plt.figure(f"50thPercentile - {kernel_used}")
    plt.plot(arcsec_radii, data_intensities, label = "data")
    plt.plot(arcsec_radii, model_intensities, label = "model")

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars}")
    plt.legend(loc = "best")
