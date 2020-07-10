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
    ranges.append([1, 7])  # Rout (arcseconds)
    ranges.append([0,  1])  # SigFrac = Sig2/Sig1
    ranges.append([0,  3])  # p2

    return ranges

def mcmc(data, nwalkers, burnin_steps, production_steps):
    print("\n----------------------------------------------------------------------------------------------------\n")
    print("   emcee - Markov chain Monte Carlo - Double model")
    print("\n----------------------------------------------------------------------------------------------------\n")

    ### Generate constants -----------------------------------------------------------------------------------------------

    inclination = data[1]["inclination"]
    eccentricity = np.sin(inclination)
    rotation = data[1]["positionAngleMin90"]

    total_data_dimension = data[0].shape[0] # pixels
    total_data_radius = total_data_dimension / 2 # pixels

    if total_data_dimension > 500:
        data_file = "highres"
    else:
        data_file = "lowres"

    model = "double"
    model_scale = 1

    total_intensity_radii = 250

    data_arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
    data_sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

    print(f"data_image dimension: {total_data_dimension} pixels")
    print(f"                    : {total_data_dimension * data_arcsec_per_pix} arcseconds\n")

    model_arcsec_per_pix = data[1]["degreesPixelScale"] * 3600 / model_scale
    model_sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2 / model_scale

    if data_file == "lowres":
        fit_radius = 5 # Arcseconds
        crop_radius = fit_radius + 2 # Arcseconds
        variance_range = [fit_radius / data_arcsec_per_pix, crop_radius / data_arcsec_per_pix]
    else:
        fit_radius = 1.5 # Arcseconds
        crop_radius = fit_radius + 0.5 # Arcseconds
        variance_range = [1.25 / data_arcsec_per_pix, 1.75 / data_arcsec_per_pix]

    data_crop_radius = int(np.ceil(crop_radius / data_arcsec_per_pix)) # pixels
    data_coords = np.linspace(-data_crop_radius, data_crop_radius, 2 * data_crop_radius) # pixels
    data_crop_radii = np.linspace(0, crop_radius / data_arcsec_per_pix, total_intensity_radii) # pixels
    data_fit_radii = np.linspace(0, fit_radius / data_arcsec_per_pix, total_intensity_radii) # pixels

    model_crop_radius = int(np.ceil(crop_radius / model_arcsec_per_pix)) # pixels
    model_coords = np.linspace(-model_crop_radius, model_crop_radius, 2 * model_crop_radius) # pixels
    model_fit_radii = np.linspace(0, fit_radius / model_arcsec_per_pix, total_intensity_radii) # pixels

    ### Get data intensity profile ----------------------------------------------------------------------------------------------

    major_axis = (data[1]["beamSemiMajor"] * 2) / data[1]["degreesPixelScale"] # pixels
    minor_axis = (data[1]["beamSemiMinor"] * 2) / data[1]["degreesPixelScale"] # pixels
    angle = data[1]['beamPA'] * (np.pi / 180) # radian (PA already defined as 90 degrees)

    print(f"beam semi major axis: {major_axis} pixels")
    print(f"                    : {major_axis * data_arcsec_per_pix} arcseconds\n")
    print(f"beam semi minor axis: {minor_axis} pixels")
    print(f"                    : {minor_axis * data_arcsec_per_pix} arcseconds\n")
    print(f"beam position angle : {angle} radian")
    print(f"                    : {data[1]['beamPA']} degrees\n")

    print("\nGenerating data intensity profiles.")
    convolved_data = convolveDataImage(data)

    print(f"\nvariance_range: {variance_range} pixels")
    data_intensities, variance = getDataIntensities(convolved_data, data_fit_radii, data_crop_radii, eccentricity, rotation, variance_range)
    print(f"variance: {variance}")

    ### Setup model -------------------------------------------------------------------------------------------------------------

    single_Rin = 0.18
    single_Rout = 0.50
    single_p = 1.5

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
    Rout = 5
    SigFrac = 0.5 # Sig2/Sig1
    p2 = 1

    fixed_pars = (v, k, i, T0, q, Sig1, Rin, R1, p1)
    free_pars = np.array([Rout, SigFrac, p2])
    free_labels = ["Rout", "SigFrac", "p2"]

    pars_ranges = parameter_ranges(R1)

    # Generate the convolution kernels for the model image
    print(f"\nGenerating model kernels")
    model_kernel = generateModelKernels(data, model_scale)

    ### Setup mcmc sampler ------------------------------------------------------------------------------------------------------

    ndim = len(free_pars)

    constants = (fixed_pars, pars_ranges, data_intensities, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, variance, model)

    with Pool() as pool:

        print(f"\nInitialize sampler with {nwalkers} walkers and {ndim} free parameters ({free_labels})")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProbability, args=[constants], pool=pool)

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
    dir_name = f"{time}_{data_file}_{model}_{nwalkers}_{burnin_steps}_{production_steps}"
    dir_path = os.path.join(results_directory, dir_name)
    os.mkdir(dir_path)
    print(f"\nCreated '{dir_name}' directory in results\n")

    # Create txt file with initial values
    with open(os.path.join(dir_path, "init.txt"), "a", encoding = "utf-8") as file:
        parameter_txt = "Choices, constants and initial parameters\n"
        parameter_txt += f"\n"
        parameter_txt += f"Free parameters        {free_labels}\n"
        parameter_txt += f"Walkers                {nwalkers}\n"
        parameter_txt += f"Burn-in step           {burnin_steps}\n"
        parameter_txt += f"Production steps       {production_steps}\n"
        parameter_txt += f"\n"
        parameter_txt += f"Image dimension        {total_data_dimension * data_arcsec_per_pix} arcseconds\n"
        parameter_txt += f"Radius of image        {total_data_radius * data_arcsec_per_pix} arcseconds\n"
        parameter_txt += f"Radius of fit          {fit_radius} arcseconds\n"
        parameter_txt += f"Radius of cropping     {crop_radius} arcseconds\n"
        parameter_txt += f"\n"
        parameter_txt += f"Model scale            {model_scale}x\n"
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
    data_image = cropImage(data_image, data_crop_radius)

    convolved_data = convolveDataImage(data)
    convolved_data[np.where(convolved_data <= 0.0)] = np.min(convolved_data[np.where(convolved_data > 0)])
    convolved_data = cropImage(convolved_data, data_crop_radius)

    model_image = getImageMatrix(fixed_pars, mcmc_pars50th, model_coords, model_arcsec_per_pix, model_sr_per_pix, model)
    model_image[np.where(model_image <= 0.0)] = np.min(model_image[np.where(model_image > 0)])

    convolved_model = convolve(model_image, model_kernel)
    convolved_model[np.where(convolved_model <= 0.0)] = np.min(convolved_model[np.where(convolved_model > 0)])

    data_dimensions = np.asarray(data_image.shape)
    data_center = data_dimensions / 2

    data_extent = [(-data_center[0]) * data_arcsec_per_pix, (data_dimensions[0] - data_center[0]) * data_arcsec_per_pix,
        (-data_center[1]) * data_arcsec_per_pix, (data_dimensions[1] - data_center[1]) * data_arcsec_per_pix]

    model_dimensions = np.asarray(model_image.shape)
    model_center = model_dimensions / 2

    model_extent = [(-model_center[0]) * model_arcsec_per_pix, (model_dimensions[0] - model_center[0]) * model_arcsec_per_pix,
        (-model_center[1]) * model_arcsec_per_pix, (model_dimensions[1] - model_center[1]) * model_arcsec_per_pix]

    fig_name = "Original data"
    plt.figure(fig_name, figsize = (5,5))
    plt.imshow(data_image, origin="lower", cmap="inferno", extent = data_extent)
    plt.xlabel("Rel. RA (arcsec)")
    plt.ylabel("Rel. Dec (arcsec)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")


    fig_name = "Convolved data"
    plt.figure(fig_name, figsize = (5,5))
    plt.imshow(convolved_data, origin="lower", cmap="inferno", extent = data_extent)
    plt.xlabel("Rel. RA (arcsec)")
    plt.ylabel("Rel. Dec (arcsec)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    fig_name = "Original model"
    plt.figure(fig_name, figsize = (5,5))
    plt.imshow(model_image, origin="lower", cmap="inferno", extent = model_extent)
    plt.xlabel("Rel. RA (arcsec)")
    plt.ylabel("Rel. Dec (arcsec)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    fig_name = "Convolved model"
    plt.figure(fig_name, figsize = (5,5))
    plt.imshow(convolved_model, origin="lower", cmap="inferno", extent = model_extent)
    plt.xlabel("Rel. RA (arcsec)")
    plt.ylabel("Rel. Dec (arcsec)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")


    fig_name = "Normal_images"
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = fig_name)

    ax1.imshow(data_image, origin="lower", cmap="inferno", extent = data_extent)
    ax1.set_title("Original data")

    ax2.imshow(model_image, origin="lower", cmap="inferno", extent = model_extent)
    ax2.set_title("Original model (50th%)")

    ax3.imshow(convolved_data, origin="lower", cmap="inferno", extent = data_extent)
    ax3.set_title("Convolved data")

    ax4.imshow(convolved_model, origin="lower", cmap="inferno", extent = model_extent)
    ax4.set_title("Convolved model (50th%)")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    fig_name = "Logarithmic_images"
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = fig_name)

    ax1.imshow(data_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = data_extent)
    ax1.set_title("Original data")

    ax2.imshow(model_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = model_extent)
    ax2.set_title("Original model (50th%)")

    ax3.imshow(convolved_data, origin="lower", norm=LogNorm(), cmap="inferno", extent = data_extent)
    ax3.set_title("Convolved data")

    ax4.imshow(convolved_model, origin="lower", norm=LogNorm(), cmap="inferno", extent = model_extent)
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
    mcmc50th_intensities = getModelIntensities(mcmc_pars50th, fixed_pars, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, model)

    percentiles = range(16,81)

    print(f"\nGenerating the {percentiles} percentile intensity profiles from the flat samples:")
    sample_intensities = []
    for percentile in tqdm(percentiles):
        pars = []
        for i in range(ndim):
            pars.append(np.percentile(flat_samples[:, i], percentile))

        sample_intensities.append(getModelIntensities(pars, fixed_pars, model_fit_radii, model_coords, model_kernel, model_arcsec_per_pix, model_sr_per_pix, model))

    data_fit_arcsec_radii = data_fit_radii * data_arcsec_per_pix
    model_fit_arcsec_radii = model_fit_radii * model_arcsec_per_pix

    # Plot data intensities
    fig_name = f"Data_intensity_profile"
    plt.figure(fig_name)

    plt.plot(data_fit_arcsec_radii, data_intensities)

    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Normalised intensity")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Plot model intensities
    fig_name = f"model_intensity_profile"
    plt.figure(fig_name)

    plt.plot(model_fit_arcsec_radii, mcmc50th_intensities)

    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Normalised intensity")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Plot normal
    fig_name = f"Normal_intensity_profile"
    plt.figure(fig_name)
    plt.title(f"Rout = {mcmc_pars50th[0]:.3f}; SigFrac = {mcmc_pars50th[1]:.3f}; p2 = {mcmc_pars50th[2]:.3f}")

    for model_intensities in sample_intensities:
        plt.plot(model_fit_arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(data_fit_arcsec_radii, data_intensities, label = "data")
    plt.plot(model_fit_arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Normalised intensity")

    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    # Plot logarithmic
    fig_name = f"Logarithmic_intensity_profile"
    plt.figure(fig_name)
    plt.title(f"Rout = {mcmc_pars50th[0]:.3f}; SigFrac = {mcmc_pars50th[1]:.3f}; p2 = {mcmc_pars50th[2]:.3f}")

    for model_intensities in sample_intensities:
        plt.plot(model_fit_arcsec_radii, model_intensities, color = "orange", alpha = 0.3)

    plt.plot(data_fit_arcsec_radii, data_intensities, label = "data")
    plt.plot(model_fit_arcsec_radii, mcmc50th_intensities, color = "red", label = "model - 50th")

    plt.xlabel("Arcseconds")
    plt.ylabel("Normalised intensity")
    plt.yscale('log')

    plt.legend(loc = "best")

    plt.savefig(os.path.join(dir_path, f"{fig_name}.png"))
    print(f"Saved {fig_name}.png")

    plt.close("all")
