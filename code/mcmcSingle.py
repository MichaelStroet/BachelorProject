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

from modelImage import *
from convolution import *
from mcmcFunctions import *

def parameter_ranges(free_pars, arcsec_per_pix):

    ranges = []
    ranges.append([0.01, 0.3]) # Rin (arcseconds)
    ranges.append([0.5, 3])     # Rout (arcseconds)
    ranges.append([0, 2])       # p

    return ranges

def mcmc(data, nwalkers, burnin_steps, production_steps):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo - Single model")
    print("\n----------------------------------------------------------------------------------------------------\n")

    ### Generate constants -----------------------------------------------------------------------------------------------

    total_intensity_radii = 100

    pixel_dimension = min(data[0].shape) # pixels
    pixel_radius = pixel_dimension / 2 # pixels
    PIXEL_COORDS = np.linspace(-pixel_radius, pixel_radius, pixel_dimension) # pixels
    PIXEL_RADII = np.linspace(0, pixel_radius, total_intensity_radii) # pixels

    arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
    sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    fit_radius = 2 # Arcseconds
    fit_radii = np.linspace(0, fit_radius, total_intensity_radii) # Arcseconds

    crop_radius = int(np.ceil((fit_radius + 0.1) / arcsec_per_pix)) # pixels

    inclination = data[1]["inclination"]
    eccentricity = np.sin(inclination)
    rotation = data[1]["positionAngleMin90"]

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    print("\nGenerating data intensity profiles.")
    FIT_DATA_INTENSITIES = getDataIntensities(data, fit_radius, eccentricity, rotation)
    TOTAL_DATA_INTENSITIES = getDataIntensities(data, pixel_radius, eccentricity, rotation)

    ### Setup model -------------------------------------------------------------------------------------------------------------

    model = "single"

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = inclination # radian
    T0 = 30 # K
    q = 0.25
    Sig0 = 0.25 # kg m^-2

    # Free parameters guesses
    Rin = 0.2 # Arcseconds
    Rout = 1  # Arcseconds
    p = 1

    fixed_pars = (v, k, i, T0, q, Sig0)
    free_pars = np.array([Rin, Rout, p])
    free_labels = ["Rin", "Rout", "p"]

    # Generate the convolution kernels for the model image
    print(f"\nGenerating model kernels")
    model_kernel_area, model_kernel_peak = generateModelKernels(data)

    MODEL_KERNEL = model_kernel_area
    kernel_used = "area"

    print(f"\nModel kernel used is: {kernel_used}")

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    ndim = len(free_pars)

    CONSTANTS = (pixel_dimension, PIXEL_COORDS, fit_radii, arcsec_per_pix, sr_per_pix, FIT_DATA_INTENSITIES, fixed_pars, MODEL_KERNEL, model, crop_radius)

    with Pool() as pool:

        print(f"\nInitialize sampler with {nwalkers} walkers and {ndim} free parameters ({free_labels})")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=[CONSTANTS, parameter_ranges], pool=pool)

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

    if pixel_dimension > 500:
        data_file = "highres"
    else:
        data_file = "lowres"

    # Create directory for saving results
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    dir_name = f"{time}_{data_file}_{model}_{nwalkers}_{burnin_steps}_{production_steps}"
    dir_path = os.path.join(results_directory, dir_name)
    os.mkdir(dir_path)
    print(f"\nCreated '{dir_name}' directory in results\n")

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
        parameter_txt += f"Image dimension        {pixel_dimension * arcsec_per_pix} arcseconds\n"
        parameter_txt += f"Radius of image        {pixel_radius * arcsec_per_pix} arcseconds\n"
        parameter_txt += f"Radius of fit          {fit_radius} arcseconds\n"
        parameter_txt += f"Radius of cropping     {crop_radius * arcsec_per_pix} arcseconds\n"
        parameter_txt += f"Arcseconds per pixel   {arcsec_per_pix} arcseconds\n"
        parameter_txt += f"\n"
        parameter_txt += f"v                      {v} Hz\n"
        parameter_txt += f"k                      {k} m^2/kg\n"
        parameter_txt += f"i                      {i} radian\n"
        parameter_txt += f"R in                   {Rin} arcseconds\n"
        parameter_txt += f"R out                  {Rout} arcseconds\n"
        parameter_txt += f"T0                     {T0} K\n"
        parameter_txt += f"q                      {q}\n"
        parameter_txt += f"Sigma0                 {Sig0} kg/m^2\n"
        parameter_txt += f"p                      {p}\n"

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
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # Visualise production steps for each parameter
    fig_name = f"Production"
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True, num = fig_name)

    fig.suptitle(f"Sampler Production steps")

    for i, label in enumerate(free_labels):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(label)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # original and convolved model image from mcmc parameters
    data_image = data[0]
    data_image[np.where(data_image <= 0.0)] = np.min(data_image[np.where(data_image > 0)])

    convolved_data = convolveDataImage(data)
    convolved_data[np.where(convolved_data <= 0.0)] = np.min(convolved_data[np.where(convolved_data > 0)])

    model_image = getImageMatrix(fixed_pars, mcmc_pars50th, PIXEL_COORDS, arcsec_per_pix, sr_per_pix, model)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])

    convolved_model = convolve(model_image, MODEL_KERNEL)
    convolved_model[np.where(convolved_model <= 0.0)] = np.min(convolved_model[np.where(convolved_model > 0)])

    centerPixel = (data[1]["xCenterPixel"], data[1]["yCenterPixel"])
    pixelDimension = data[0].shape

    extent = [(-centerPixel[0]) * arcsec_per_pix, (pixelDimension[0] - centerPixel[0]) * arcsec_per_pix,
        (-centerPixel[1]) * arcsec_per_pix, (pixelDimension[1] - centerPixel[1]) * arcsec_per_pix]

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
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # Corner plot
    fig_name = f"Corner_plot"
    fig = corner.corner(flat_samples, labels=free_labels, truths=free_pars, show_titles=True, quantiles=[0.16, 0.50, 0.84])
    fig.canvas.set_window_title(fig_name)

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # Intensity profile comparison
    mcmc50th_intensities = getModelIntensities(mcmc_pars50th, PIXEL_COORDS, PIXEL_RADII, arcsec_per_pix, sr_per_pix, fixed_pars, MODEL_KERNEL, model, crop_radius)

    model_plots = 5
    sample_indeces = np.random.randint(0, production_steps, model_plots)

    print(f"\nGenerating {model_plots} intensity profiles from the flat samples:")
    sample_intensities = []
    for i in tqdm(sample_indeces):
        sample_intensities.append(getModelIntensities(flat_samples[i], PIXEL_COORDS, PIXEL_RADII, arcsec_per_pix, sr_per_pix, fixed_pars, MODEL_KERNEL, model, crop_radius))

    arcsec_radius = pixel_radius * arcsec_per_pix
    arcsec_radii = np.linspace(0, arcsec_radius, total_intensity_radii)

    # Plot normal
    fig_name = f"Intensity_profile"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Intensity [Arbitrary units]")

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # Plot logarithmic
    fig_name = f"Intensity_profile_log"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Intensity [Arbitrary units]")
    plt.yscale('log')

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    #plt.clf()
    print(f"Saved {fig_name}.png")

    # Plot logarithmic
    fig_name = f"Intensity_profile_fitting_area_log"
    plt.figure(fig_name)

    for model_intensities in sample_intensities:
        plt.plot(arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, label = "data")
    plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.xlim([0, fit_radius])

    plt.ylabel("Intensity [Arbitrary units]")
    plt.yscale('log')
    plt.ylim([10e-10, 10])

    plt.title(f"fixed: {fixed_pars}\nmcmc: {mcmc_pars50th}")
    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))

    print(f"Saved {fig_name}.png")

    plt.close("all")

    # ### Testing p values
    #
    # data_intensities = TOTAL_DATA_INTENSITIES
    #
    # fig_name = f"p_values_1"
    # plt.figure(fig_name)
    #
    # plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, color = "black", label = "data")
    # plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")
    #
    # p_values = np.linspace(0, 2, 10)
    # print("Plotting p value intensities1")
    # for p in tqdm(p_values):
    #     model_pars = mcmc_pars50th
    #     model_pars[2] = p
    #
    #     model_intensities = getModelIntensities(model_pars, PIXEL_COORDS, PIXEL_RADII, arcsec_per_pix, sr_per_pix, fixed_pars, MODEL_KERNEL, model, crop_radius)
    #     plt.plot(arcsec_radii, model_intensities, label = f"p = {p:.2f}")
    #
    # plt.xlabel("Arcseconds")
    # plt.ylabel("Intensity [Arbitrary units]")
    # plt.yscale("log")
    #
    # plt.xlim([0,4])
    # plt.ylim([1e-3, 1])
    #
    # plt.legend(loc = "best")
    #
    # plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    # print(f"Saved {fig_name}.png")
    #
    # fig_name = f"p_values_2"
    # plt.figure(fig_name)
    #
    # plt.plot(arcsec_radii, TOTAL_DATA_INTENSITIES, color = "black", label = "data")
    # plt.plot(arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")
    #
    # p_values = np.linspace(0, 2, 10)
    # print("Plotting p value intensities2")
    # for p in tqdm(p_values):
    #     model_pars = mcmc_pars50th
    #     model_pars[2] = p
    #
    #     model_intensities = getModelIntensities(model_pars, PIXEL_COORDS, PIXEL_RADII, arcsec_per_pix, sr_per_pix, fixed_pars, MODEL_KERNEL, model, crop_radius)
    #     plt.plot(arcsec_radii, model_intensities, label = f"p = {p:.2f}")
    #
    # plt.xlabel("Arcseconds")
    # plt.ylabel("Intensity [Arbitrary units]")
    # plt.yscale("log")
    #
    # plt.xlim([0,4])
    # plt.ylim([1e-2, 10])
    #
    # plt.legend(loc = "best")
    #
    # plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    # print(f"Saved {fig_name}.png")
    #
    # plt.show()
