# Michael Stroet  11293284

import emcee
import corner
import os, sys
import numpy as np
import matplotlib.pyplot as plt

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
results_directory = os.path.join(data_directory, "results")

from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from matplotlib.colors import LogNorm
from astropy.convolution import convolve
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fitsFiles import *
from modelImage import *
from convolution import *
from radialProfileCircle import getCircleProfile
from radialProfileEllipse import getEllipseProfile

def getDataIntensities(data, max_radius, eccentricity, rotation):

    convolved_image = convolveDataImage(data)

    data_intensities = np.asarray(getEllipseProfile(convolveDataImage(data), max_radius, eccentricity, rotation))
    data_max = np.max(data_intensities)

    # Return the intensities scaled in such a way that the peak is at 1
    return data_intensities / data_max

def getModelIntensities(free_pars, PIXEL_COORDS, radii, SR_PER_PIX, fixed_pars, MODEL_KERNEL, model):

    # generate the model image
    model_image = getImageMatrix(fixed_pars, free_pars, PIXEL_COORDS, SR_PER_PIX, model)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, MODEL_KERNEL)

    # Generate a radial intensity profile from the model image
    model_intensities = np.asarray(getCircleProfile(convolved_model_image, radii))
    model_max = np.max(model_intensities)

    if model_max == np.nan or model_max < 0:
        print(f"model_max: {model_max}")

    # Return the intensities scaled in such a way that the peak is at 1
    return model_intensities / model_max

def parameter_ranges(free_pars, ARCSEC_PER_PIX):
    Rin, Rout, SigFrac, R1, p1, p2, q = free_pars

    ranges = []
    ranges.append([0.1 / ARCSEC_PER_PIX, Rout]) # Rin (pixels)
    ranges.append([2*Rin, 30 / ARCSEC_PER_PIX]) # Rout (pixels)
    ranges.append([0, 10])                      # SigFrac = Sig2/Sig1
    ranges.append([Rin, Rout])                      # R1 (pixels)
    ranges.append([0, 20])                      # p1
    ranges.append([0, 20])                      # p2
    ranges.append([0, 20])                      # q

    return ranges

# log-prior function
def logPrior(free_pars, ARCSEC_PER_PIX):

    for parameter, range in zip(free_pars, parameter_ranges(free_pars, ARCSEC_PER_PIX)):
        if not range[0] <= parameter <= range[1]:
            return -np.inf

    return 0

# log-likelihood function
def logLikelihood(free_pars, CONSTANTS):
    PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, fixed_pars, MODEL_KERNEL, model = CONSTANTS

    I_model = getModelIntensities(free_pars, PIXEL_COORDS, FIT_RADII, SR_PER_PIX, fixed_pars, MODEL_KERNEL, model)
    variance = np.sum((FIT_DATA_INTENSITIES - I_model)**2) / len(FIT_DATA_INTENSITIES)

    log_likelihood = -0.5 * np.sum(((FIT_DATA_INTENSITIES - I_model)**2 / variance) + np.log(2 * np.pi * variance))

    return log_likelihood

# log-probability function
def logProbability(free_pars, CONSTANTS):
    PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, fixed_pars, MODEL_KERNEL, model = CONSTANTS

    log_prior = logPrior(free_pars, ARCSEC_PER_PIX)
    if np.isfinite(log_prior):
        log_likelihood = logLikelihood(free_pars, CONSTANTS)
        if np.isnan(log_likelihood):
            print(f"log_likelihood returned nan, return -infinity")
            return -np.inf

        return log_likelihood + log_prior
    else:
        return log_prior

def mcmc(data, nwalkers, burnin_steps, production_steps):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo - Double model")
    print("\n----------------------------------------------------------------------------------------------------\n")

    ### Generate constants -----------------------------------------------------------------------------------------------

    total_intensity_radii = 100

    PIXEL_DIMENSION = min(data[0].shape)
    PIXEL_RADIUS = PIXEL_DIMENSION / 2
    PIXEL_COORDS = np.linspace(-PIXEL_RADIUS, PIXEL_RADIUS, PIXEL_DIMENSION)
    PIXEL_RADII = np.linspace(0, PIXEL_RADIUS, total_intensity_radii)

    ARCSEC_PER_PIX = data[1]["degreesPixelScale"] * 3600
    SR_PER_PIX = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    FIT_RADIUS = 5 / ARCSEC_PER_PIX
    FIT_RADII = np.linspace(0, FIT_RADIUS, total_intensity_radii)

    inclination = data[1]["inclination"]
    eccentricity = np.sin(inclination)
    rotation = data[1]["positionAngleMin90"]

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    FIT_DATA_INTENSITIES = getDataIntensities(data, FIT_RADIUS, eccentricity, rotation)
    TOTAL_DATA_INTENSITIES = getDataIntensities(data, PIXEL_RADIUS, eccentricity, rotation)

    ### Setup model -------------------------------------------------------------------------------------------------------------

    model = "double"

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = inclination # radian
    T0 = 30 # K
    Sig1 = 0.25 # kg m^-2

    # Free parameters guesses
    Rin = 1 #10 # Pixels
    Rout = 25 #200 # Pixels
    SigFrac = 3 # Sig2/Sig1
    R1 = (Rin + Rout) / 2 # Pixels
    p1 = 5
    p2 = 5
    q = 5

    fixed_pars = (v, k, i, T0, Sig1)
    free_pars = np.array([Rin, Rout, SigFrac, R1, p1, p2, q])

    free_labels = ["Rin", "Rout", "SigFrac", "R1", "p1", "p2", "q"]

    # Generate the convolution kernels for the model image
    model_kernel_area, model_kernel_peak = generateModelKernels(data)

    MODEL_KERNEL = model_kernel_area
    kernel_used = "area"

    print(f"\nModel kernel used is: {kernel_used}")

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    ndim = len(free_pars)

    CONSTANTS = (PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, fixed_pars, MODEL_KERNEL, model)

    with Pool() as pool:

        print(f"\nInitialize sampler with {nwalkers} walkers and {ndim} free parameters ({free_labels})")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=[CONSTANTS], pool=pool)

        ### Burn-in sampler ---------------------------------------------------------------------------------------------------------

        start_locations = free_pars + (1e-3 * free_pars) * np.random.randn(nwalkers, ndim)


        print(f"\nRunning {burnin_steps} burn-in steps:")
        burnin_state = sampler.run_mcmc(start_locations, burnin_steps, progress = True)

        burnin_samples = sampler.get_chain(flat=False)
        sampler.reset()

        ### Production run ----------------------------------------------------------------------------------------------------------

        print(f"\nRunning {production_steps} production steps:")
        sampler.run_mcmc(burnin_state, production_steps, progress = True)

    samples = sampler.get_chain(flat = False)
    flat_samples = sampler.get_chain(flat=True)

    # Get the 50th percentile parameter values
    mcmc_pars50th = []
    for i in range(ndim):
        mcmc_pars50th.append(np.percentile(flat_samples[:, i], 50))

    ### Visualise results -------------------------------------------------------------------------------------------------------

    # Create directory for saving results
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dir_name = f"{time}_{model}_{nwalkers}_{ndim}_{burnin_steps}_{production_steps}"
    dir_path = os.path.join(results_directory, dir_name)
    os.mkdir(dir_path)

    # Create txt file with initial values
    with open(os.path.join(dir_path, "init.txt"), "a", encoding = "utf-8") as file:
        parameter_txt = "Choices, constants and initial parameters\n"
        parameter_txt += f"\n"
        parameter_txt += f"Convolution kernel     {kernel_used} normalised\n"
        parameter_txt += f"Free parameters        {free_labels}\n"
        parameter_txt += f"Walkers                {nwalkers}\n"
        parameter_txt += f"Burn-in step           {burnin_steps}\n"
        parameter_txt += f"Production steps       {production_steps}\n"
        parameter_txt += f"\n"
        parameter_txt += f"Image dimension        {PIXEL_DIMENSION} pixels\n"
        parameter_txt += f"Radius of image        {PIXEL_RADIUS} pixels\n"
        parameter_txt += f"Radius of fit          {FIT_RADIUS} pixels\n"
        parameter_txt += f"Arcseconds per pixel   {ARCSEC_PER_PIX} arcsec\n"
        parameter_txt += f"\n"
        parameter_txt += f"v                      {v} Hz\n"
        parameter_txt += f"k                      {k} m^2/kg\n"
        parameter_txt += f"i                      {i} radian\n"
        parameter_txt += f"R in                   {Rin} pixels\n"
        parameter_txt += f"R out                  {Rout} pixels\n"
        parameter_txt += f"Sig2 / Sig1            {SigFrac} K\n"
        parameter_txt += f"R1                     {R1} pixels\n"
        parameter_txt += f"p1                     {p1}\n"
        parameter_txt += f"p2                     {p2}\n"
        parameter_txt += f"q                      {q}\n"

        file.write(parameter_txt)

    # Visualise burnin steps for each parameter
    fig_name = f"Burn-in"
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True, num = fig_name)
    fig.suptitle(f"Sampler Burn-in steps")

    for i, label in enumerate(free_labels):
        ax = axes[i]
        ax.plot(burnin_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(burnin_samples))
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # original and convolved model image from mcmc parameters
    data_image = data[0]
    data_image[np.where(data_image <= 0.0)] = np.min(data_image[np.where(data_image > 0)])

    convolved_data = convolveDataImage(data)
    convolved_data[np.where(convolved_data <= 0.0)] = np.min(convolved_data[np.where(convolved_data > 0)])

    model_image = getImageMatrix(fixed_pars, mcmc_pars50th, PIXEL_COORDS, SR_PER_PIX, model)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])

    convolved_model = convolve(model_image, MODEL_KERNEL)
    convolved_model[np.where(convolved_model <= 0.0)] = np.min(convolved_model[np.where(convolved_model > 0)])

    centerPixel = (data[1]["xCenterPixel"], data[1]["yCenterPixel"])
    pixelDimension = data[0].shape

    extent = [(-centerPixel[0]) * ARCSEC_PER_PIX, (pixelDimension[0] - centerPixel[0]) * ARCSEC_PER_PIX,
        (-centerPixel[1]) * ARCSEC_PER_PIX, (pixelDimension[1] - centerPixel[1]) * ARCSEC_PER_PIX]

    fig_name = "Images"
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = fig_name)

    ax1.imshow(data_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax1.set_title("Original data")

    ax2.imshow(model_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax2.set_title("Original model (50th%)")

    ax3.imshow(convolved_data, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax3.set_title("Convolved data")

    ax4.imshow(convolved_model, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax4.set_title("Convolved model (50th%)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Corner plot
    fig_name = f"Corner_plot"
    fig = corner.corner(flat_samples, labels=free_labels, truths=free_pars, show_titles=True, quantiles=[0.16, 0.50, 0.84])
    fig.canvas.set_window_title(fig_name)

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Intensity profile comparison
    mcmc50th_intensities = getModelIntensities(mcmc_pars50th, PIXEL_COORDS, PIXEL_RADII, SR_PER_PIX, fixed_pars, MODEL_KERNEL, model)

    model_plots = 25
    sample_indeces = np.random.randint(0, production_steps, model_plots)

    print(f"\nGenerating {model_plots} intensity profiles from the flat samples:")
    sample_intensities = []
    for i in tqdm(sample_indeces):
        sample_intensities.append(getModelIntensities(flat_samples[i], PIXEL_COORDS, PIXEL_RADII, SR_PER_PIX, fixed_pars, MODEL_KERNEL, model))

    arcsec_radius = PIXEL_RADIUS * ARCSEC_PER_PIX
    arcsec_radii = np.linspace(0, arcsec_radius, total_intensity_radii)

    # Plot normal
    fig_name = f"Intensity_profile"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Intensity [Jy/beam]")

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Plot logarithmic
    fig_name = f"Intensity_profile_log"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Intensity [Jy/beam]")
    plt.yscale('log')

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Plot logarithmic
    fig_name = f"Intensity_profile_fitting_area_log"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.xlim([0, FIT_RADIUS * ARCSEC_PER_PIX])

    plt.ylabel("Intensity [Jy/beam]")
    plt.yscale('log')
    plt.ylim([10e-10, 10])

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")
