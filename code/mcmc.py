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

def getDataIntensities(data, radii):

    convolved_image = convolveDataImage(data)

    data_intensities = np.asarray(getCircleProfile(convolveDataImage(data), radii))
    data_max = np.max(data_intensities)

    # Return the intensities scaled in such a way that the peak is at 1
    return data_intensities / data_max

def getModelIntensities(free_pars, PIXEL_COORDS, radii, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL):

    # generate the model image
    model_image = getImageMatrix(FIXED_PARS, free_pars, PIXEL_COORDS, SR_PER_PIX)

    # Convolve model with the combined "round" kernel
    convolved_model_image = convolve(model_image, MODEL_KERNEL)

    # Generate a radial intensity profile from the model image
    model_intensities = np.asarray(getCircleProfile(convolved_model_image, radii))
    model_max = np.max(model_intensities)

    # Return the intensities scaled in such a way that the peak is at 1
    return model_intensities / model_max

def parameter_ranges(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL):
    Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0 = free_pars
    # PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL = GLOBAL

    ranges = []
    ranges.append([0.1 / ARCSEC_PER_PIX, Rout]) # Rin (pixels)
    ranges.append([Rin, 30 / ARCSEC_PER_PIX])   # Rout (pixels)
    ranges.append([10,50])                      # T0 (K)
    ranges.append([Rin,Rout])                   # R_br (pixels)
    ranges.append([0,p1])                       # p0
    ranges.append([p0,10])                      # p1
    ranges.append([0,10])                       # Sig0 (kg m^-2)
    ranges.append([0,10])                       # q0
    ranges.append([0,10])                      # q1
    ranges.append([Rin, Rout])                  # R0 (pixels)

    return ranges

# log-prior function
def logPrior(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL):
    # PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL = GLOBAL

    for parameter, range in zip(free_pars, parameter_ranges(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL)):
        if not range[0] <= parameter <= range[1]:
            return -np.inf

    return 0

# log-likelihood function
def logLikelihood(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL):
    # PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL = GLOBAL

    I_model = getModelIntensities(free_pars, PIXEL_COORDS, FIT_RADII, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL)
    variance = np.sum((FIT_DATA_INTENSITIES - I_model)**2) / len(FIT_DATA_INTENSITIES)

    log_likelihood = -0.5 * np.sum(((FIT_DATA_INTENSITIES - I_model)**2 / variance) + np.log(2 * np.pi * variance))

    return log_likelihood

# log-probability function
def logProbability(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL):
    # PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL = GLOBAL

    log_prior = logPrior(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL)
    if np.isfinite(log_prior):
        log_likelihood = logLikelihood(free_pars, PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL)
        if np.isnan(log_likelihood):
            print(f"why da fuck is it not a number? {log_likelihood}")

        return log_likelihood + log_prior
    else:
        return log_prior

def mcmc(data):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo")
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

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    FIT_DATA_INTENSITIES = getDataIntensities(data, FIT_RADII)
    TOTAL_DATA_INTENSITIES = getDataIntensities(data, PIXEL_RADII)

    ### Setup model -------------------------------------------------------------------------------------------------------------

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = 42.46 * (np.pi / 180) # radian

    # Free parameters guesses
    Rin = 1 #10 # Pixels
    Rout = 10 #200 # Pixels
    T0 = 27 # K
    R_br = (Rin + Rout) / 2 # Pixels
    p0 = 0.2
    p1 = 8.0
    Sig0 = 0.04 # kg m^-2
    q0 = 3.6
    q1 = 1
    R0 = (Rin + Rout) / 2 # Pixels --> radius units dont matter, only their ratio

    # global FIXED_PARS
    FIXED_PARS = (v, k, i)
    free_pars = np.array([Rin, Rout, T0, R_br, p0, p1, Sig0, q0, q1, R0])
    free_labels = ["Rin", "Rout", "T0", "R_br", "p0", "p1", "Sig0", "q0", "q1", "R0"]

    # Generate the convolution kernels for the model image
    model_kernel_area, model_kernel_peak = generateModelKernels(data)

    MODEL_KERNEL = model_kernel_area
    kernel_used = "area"

    print(f"\nModel kernel used is: {kernel_used}")

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    nwalkers = 25
    ndim = len(free_pars)

    CONSTANTS = (PIXEL_DIMENSION, PIXEL_COORDS, FIT_RADII, ARCSEC_PER_PIX, SR_PER_PIX, FIT_DATA_INTENSITIES, FIXED_PARS, MODEL_KERNEL)

    with Pool() as pool:

        print(f"\nInitialize sampler with {nwalkers} walkers and {ndim} free parameters ({free_labels})")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=CONSTANTS, pool=pool)

        ### Burn-in sampler ---------------------------------------------------------------------------------------------------------

        start_locations = free_pars + (1e-3 * free_pars) * np.random.randn(nwalkers, ndim)
        burnin_steps = 20

        print(f"\nRunning {burnin_steps} burn-in steps:")
        burnin_state = sampler.run_mcmc(start_locations, burnin_steps, progress = True)

        burnin_samples = sampler.get_chain(flat=False)
        sampler.reset()

        ### Production run ----------------------------------------------------------------------------------------------------------

        production_steps = 20

        print(f"\nRunning {production_steps} production steps:")
        sampler.run_mcmc(burnin_state, production_steps, progress = True)

    samples = sampler.get_chain(flat = False)
    print(f"samples ({samples.shape}):\n{samples}")

    flat_samples = sampler.get_chain(flat=True)
    print(f"flat_samples ({flat_samples.shape}):\n{flat_samples}")
    # Get the 50th percentile parameter values
    mcmc_pars50th = []
    for i in range(ndim):
        mcmc_pars50th.append(np.percentile(flat_samples[:, i], 50))

    ### Visualise results -------------------------------------------------------------------------------------------------------

    # Create directory for saving results
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dir_name = f"{time}_{kernel_used}_{nwalkers}_{ndim}_{burnin_steps}_{production_steps}"
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
        parameter_txt += f"T0                     {T0} K\n"
        parameter_txt += f"R break                {R_br} pixels\n"
        parameter_txt += f"p0                     {p0}\n"
        parameter_txt += f"p1                     {p1}\n"
        parameter_txt += f"Sigma0                 {Sig0} kg/m^2\n"
        parameter_txt += f"q0                     {q0}\n"
        parameter_txt += f"q1                     {q1}\n"
        parameter_txt += f"R0                     {R0} pixels\n"

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

    model_image = getImageMatrix(FIXED_PARS, mcmc_pars50th, PIXEL_COORDS, SR_PER_PIX)
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
    mcmc50th_intensities = getModelIntensities(mcmc_pars50th, PIXEL_COORDS, PIXEL_RADII, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL)

    model_plots = 25
    sample_indeces = np.random.randint(0, production_steps, model_plots)

    print(f"\nGenerating {model_plots} intensity profiles from the flat samples:")
    sample_intensities = []
    for i in tqdm(sample_indeces):
        sample_intensities.append(getModelIntensities(flat_samples[i], PIXEL_COORDS, PIXEL_RADII, SR_PER_PIX, FIXED_PARS, MODEL_KERNEL))

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

    plt.title(f"fixed: {FIXED_PARS}\nmcmc: {mcmc_pars50th}")
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

    plt.title(f"fixed: {FIXED_PARS}\nmcmc: {mcmc_pars50th}")
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

    plt.title(f"fixed: {FIXED_PARS}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    ### testing plots

    print(f"\nTotal data intensities:\n{TOTAL_DATA_INTENSITIES}")
    print(f"\mcmc50th_intensities:\n{mcmc50th_intensities}")

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 10))

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    img = ax1.imshow(convolved_data, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
    ax1.set_title("Convolved data")

    ax2.plot(PIXEL_RADII, TOTAL_DATA_INTENSITIES)
    ax2.plot(PIXEL_RADII, mcmc50th_intensities)
    ax2.set_yscale("log")

    fig.colorbar(img, cax=cax, orientation='vertical')
