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

def parameter_ranges(R1):

    ranges = []
    ranges.append([R1, 5])  # Rout (arcseconds)
    ranges.append([0,  2])  # SigFrac = Sig2/Sig1
    ranges.append([0,  3])  # p2

    return ranges

def mcmc(data, nwalkers, burnin_steps, production_steps):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo - Double model")
    print("\n----------------------------------------------------------------------------------------------------\n")

    ### Generate constants -----------------------------------------------------------------------------------------------

    total_intensity_radii = 250

    pixel_dimension = min(data[0].shape) # pixels
    pixel_radius = pixel_dimension / 2 # pixels
    pixel_coords = np.linspace(-pixel_radius, pixel_radius, pixel_dimension) # pixels
    pixel_radii = np.linspace(0, pixel_radius, total_intensity_radii) # pixels

    arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
    sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    fit_radius = 2 # Arcseconds
    fit_radii = np.linspace(0, fit_radius, total_intensity_radii) # Arcseconds

    crop_radius = int(np.ceil(fit_radius + (1 / arcsec_per_pix)))

    inclination = data[1]["inclination"]
    eccentricity = np.sin(inclination)
    rotation = data[1]["positionAngleMin90"]

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    print("\nGenerating data intensity profiles.")
    FIT_DATA_INTENSITIES = getDataIntensities(data, fit_radii, eccentricity, rotation)
    TOTAL_DATA_INTENSITIES = getDataIntensities(data, pixel_radii, eccentricity, rotation)

    ### Setup model -------------------------------------------------------------------------------------------------------------

    model = "double"

    single_Rin = 0.11
    single_Rout = 0.89
    single_p = 0.81

    # Fixed parameters
    v = 225e9 # Hz (219-235 GHz)
    k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
    i = inclination # radian
    T0 = 30 # K
    q = 0.25
    Sig1 = 0.25 # kg m^-2
    Rin = single_Rin
    R1 = single_Rout
    p1 = single_p

    # Free parameters
    Rout = 3
    SigFrac = 1 # Sig2/Sig1
    p2 = 1

    fixed_pars = (v, k, i, T0, q, Sig1, Rin, R1, p1)
    free_pars = np.array([Rout, SigFrac, p2])
    free_labels = ["Rout", "SigFrac", "p2"]

    pars_ranges = parameter_ranges(R1)

    # Generate the convolution kernels for the model image
    model_kernel_area, model_kernel_peak = generateModelKernels(data)

    model_kernel = model_kernel_area
    kernel_used = "area"

    print(f"\nModel kernel used is: {kernel_used}")

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    ndim = len(free_pars)

    CONSTANTS = (pars_ranges, pixel_dimension, pixel_coords, fit_radii, arcsec_per_pix, sr_per_pix, FIT_DATA_INTENSITIES, fixed_pars, model_kernel, model, crop_radius)

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
        parameter_txt += f"R1                     {R1} arcseconds\n"
        parameter_txt += f"R out                  {Rout} arcseconds\n"
        parameter_txt += f"T0                     {T0} K\n"
        parameter_txt += f"q                      {q}\n"
        parameter_txt += f"SigmaFrac              {SigFrac} Sig2/Sig1\n"
        parameter_txt += f"Sigma1                 {Sig1} kg/m^2\n"
        parameter_txt += f"Sigma2                 {SigFrac * Sig1} kg/m^2\n"
        parameter_txt += f"p1                     {p1}\n"
        parameter_txt += f"p2                     {p2}\n"
        parameter_txt += f"\n"
        parameter_txt += f"Free parameters:\n{free_labels}\n"
        parameter_txt += f"Free parameter ranges:\n"

        for label, pars_range in zip(free_labels, pars_ranges):
            parameter_txt += f"{label}:   {pars_range}\n"

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
    print(f"Saved {fig_name}.png")

    # original and convolved model image from mcmc parameters
    data_image = data[0]
    data_image[np.where(data_image <= 0.0)] = np.min(data_image[np.where(data_image > 0)])

    convolved_data = convolveDataImage(data)
    convolved_data[np.where(convolved_data <= 0.0)] = np.min(convolved_data[np.where(convolved_data > 0)])

    model_image = getImageMatrix(fixed_pars, mcmc_pars50th, pixel_coords, arcsec_per_pix, sr_per_pix, model)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])

    convolved_model = convolve(model_image, model_kernel)
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
    print(f"Saved {fig_name}.png")

    # Corner plot
    fig_name = f"Corner_plot"
    fig = corner.corner(flat_samples, labels=free_labels, truths=free_pars, show_titles=True, quantiles=[0.16, 0.50, 0.84])
    fig.canvas.set_window_title(fig_name)

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Intensity profile comparison
    mcmc50th_intensities = getModelIntensities(mcmc_pars50th, pixel_coords, pixel_radii, arcsec_per_pix, sr_per_pix, fixed_pars, model_kernel, model, crop_radius)

    model_plots = 5
    sample_indeces = np.random.randint(0, production_steps, model_plots)

    print(f"\nGenerating {model_plots} intensity profiles from the flat samples:")
    sample_intensities = []
    for i in tqdm(sample_indeces):
        sample_intensities.append(getModelIntensities(flat_samples[i], pixel_coords, pixel_radii, arcsec_per_pix, sr_per_pix, fixed_pars, model_kernel, model, crop_radius))

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
