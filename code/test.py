# Michael Stroet  11293284

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

from fitsFiles import *
from modelImage import *
from convolution import *
from astropy.convolution import convolve
from radialProfileCircle import getCircleProfile
from radialProfileEllipse import getEllipseProfile

def pyName():
    return __file__.split("\\")[-1].replace(".py", "")

root_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(root_directory, "data")
ALMA_directory = os.path.join(data_directory, "ALMA-HD100546")

degreesToRadian = (np.pi / 180)
HD100546_i = 42.46 * degreesToRadian #radian
HD100546_PA = 139.1 * degreesToRadian #radian

ALMA_filenames = os.listdir(ALMA_directory)
relevant_headers = ["BMAJ", "BMIN", "BPA", "CRPIX1", "CRPIX2", "CDELT2"]
descriptions = ["beamSemiMajor", "beamSemiMinor", "beamPA", "xCenterPixel", "yCenterPixel", "degreesPixelScale"]

ALMA_data = getFitsData(ALMA_filenames, ALMA_directory, relevant_headers, descriptions)
# printFitsData(ALMA_filenames, ALMA_data)

file_index = 0
data = ALMA_data[file_index]

# Set negative noise from the image to zero
data[0][np.where(data[0] < 0.0)] = 0.0

# Add the inclination and corrected position angle to the header
data[1]["inclination"] = HD100546_i
data[1]["positionAngleMin90"] = HD100546_PA - (90 * degreesToRadian)

model_kernel_area, model_kernel_peak = generateModelKernels(data)

print("\n----------------------------------------------------------------------------------------------------\n")

# Fixed parameters
v = 225e9 # Hz (219-235 GHz)
k = 0.21 # m^2/kg (linearly scaled from 0.34 @ 365.5 GHz)
i = HD100546_i # radian
T0 = 30 # K
Sig0 = 0.25 # kg m^-2
q = 0.24

# Free parameters guesses
Rin = 1 # arcseconds
Rout = 5 # arcseconds
p = 1.5

fixed_pars = (v, k, i, T0, q, Sig0)
free_pars = np.array([Rin, Rout, p])

arcsec_per_pix = data[1]["degreesPixelScale"] * 3600
sr_per_pix = (data[1]["degreesPixelScale"] * np.pi / 180)**2

model = "single"

pixel_dimension = min(data[0].shape)
pixel_radius = pixel_dimension / 2
pixel_coords = np.linspace(-pixel_radius, pixel_radius, pixel_dimension)

total_intensity_radii = 100
crop_radius = int(np.ceil(6 / arcsec_per_pix))
intensity_radii = np.linspace(0, crop_radius, total_intensity_radii)

arcsec_radius = crop_radius * arcsec_per_pix
arcsec_radii = np.linspace(0, arcsec_radius, total_intensity_radii)

image = getImageMatrix(fixed_pars, free_pars, pixel_coords, arcsec_per_pix, sr_per_pix, model)
image_cropped = cropImage(image, crop_radius)

start_time = time.time()
convolved_image = convolve(image, model_kernel_area)
elapsed_time = time.time() - start_time
print(f"\nNormal convolution took {elapsed_time:.2f} seconds")

start_time = time.time()
convolved_cropped = convolve(image_cropped, model_kernel_area)
elapsed_time = time.time() - start_time
print(f"Cropped convolution took {elapsed_time:.2f} seconds\n")

normal_intensities = np.asarray(getCircleProfile(convolved_image, intensity_radii))
cropped_intensities = np.asarray(getCircleProfile(convolved_cropped, intensity_radii))

image_min = np.min(image[np.where(image > 0)])
image_max = np.max(image[np.where(image > 0)])
convolved_image_min = np.min(convolved_image[np.where(convolved_image > 0)])
convolved_image_max = np.max(convolved_image[np.where(convolved_image > 0)])

vmin = np.min([image_min, convolved_image_min])
vmax = np.max([image_max, convolved_image_max])

centerPixel = (data[1]["xCenterPixel"], data[1]["yCenterPixel"])
pixelDimension = data[0].shape

extent = [(-centerPixel[0]) * arcsec_per_pix, (pixelDimension[0] - centerPixel[0]) * arcsec_per_pix,
    (-centerPixel[1]) * arcsec_per_pix, (pixelDimension[1] - centerPixel[1]) * arcsec_per_pix]

cropped_dimension = np.asarray(image_cropped.shape)
cropped_center = cropped_dimension / 2

extent_cropped = [(-cropped_center[0]) * arcsec_per_pix, (cropped_dimension[0] - cropped_center[0]) * arcsec_per_pix,
    (-cropped_center[1]) * arcsec_per_pix, (cropped_dimension[1] - cropped_center[1]) * arcsec_per_pix]

### -----------------------------------------------------------------------------------------------------------------------------

fig_name = "model images"
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = fig_name)

ax1.imshow(image, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, extent = extent)
ax1.set_title("Model normal")

ax2.imshow(convolved_image, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax, extent = extent)
ax2.set_title("Convolved normal")

ax3.imshow(image, origin="lower", norm=LogNorm(), cmap="inferno", vmin=vmin, vmax=vmax, extent = extent)
ax3.set_title("Model logarithmic")

ax4.imshow(convolved_image, origin="lower", norm=LogNorm(), cmap="inferno", vmin=vmin, vmax=vmax, extent = extent)
ax4.set_title("Convolved logarithmic")

### -----------------------------------------------------------------------------------------------------------------------------

fig_name = "cropped model images"
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize = (12,12), num = fig_name)

ax1.imshow(image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
ax1.set_title("Model image")

ax2.imshow(image_cropped, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent_cropped)
ax2.set_title("Cropped model image")

ax3.imshow(convolved_image, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent)
ax3.set_title("Convolved model image")

ax4.imshow(convolved_cropped, origin="lower", norm=LogNorm(), cmap="inferno", extent = extent_cropped)
ax4.set_title("Convolved cropped image")

### -----------------------------------------------------------------------------------------------------------------------------

fig_name = f"Intensity comparison"
plt.figure(fig_name)

plt.plot(arcsec_radii, normal_intensities, label = "normal")
plt.plot(arcsec_radii, cropped_intensities, color = "red", label = f"cropped at {arcsec_radius:.2f}``")

plt.xlabel("Arcseconds")
plt.ylabel("Intensity [Jy/beam]")

plt.title(f"Normal vs cropped intensity profile")
plt.legend(loc = "best")

### -----------------------------------------------------------------------------------------------------------------------------

fig_name = f"Intensity comparison logarithmic"
plt.figure(fig_name)

plt.plot(arcsec_radii, normal_intensities, label = "normal")
plt.plot(arcsec_radii, cropped_intensities, color = "red", label = f"cropped at {arcsec_radius:.2f}``")

plt.xlabel("Arcseconds")
plt.ylabel("Intensity [Jy/beam]")
plt.yscale("log")

plt.title(f"Normal vs cropped intensity profile")
plt.legend(loc = "best")

plt.show()
