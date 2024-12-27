import numpy as np
from matplotlib import pyplot as plt

from asi import path_config
from asi.draw import draw_multi_crosss, reconstruct_rgb_envi, select_area
from asi.io.load_envi import load_spectral_image
from asi.utils import get_wavelengths

specim_iq_root = path_config.measurements / "Session1" / "SpecimIQ"

path_404 = specim_iq_root / "404" / "capture"

# White correction with small reference
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

image_path = path_404 / "404"
spectral_image, envi_header = load_spectral_image(image_path)

white_pos_list = [
    (slice(65, 102), slice(332, 370)),
    (slice(400, 430), slice(370, 400)),
]

original_rgb = reconstruct_rgb_envi(spectral_image, envi_header)
for white_pos in white_pos_list:
    original_rgb = select_area(original_rgb, white_pos)

original_rgb *= 1.5
original_rgb = original_rgb.clip(0, 1)

axes[0].imshow(original_rgb)
axes[0].set_title("Before white correction")

# White correction with selected area
white_sq_list = []
for white_pos in white_pos_list:
    white_sq = spectral_image[white_pos]
    a, b, band = white_sq.shape
    white_sq = white_sq.reshape(a * b, band)

    # replace nonzero elements with minimum value
    nonzero_elements = white_sq[white_sq != 0]
    min_elm = nonzero_elements.min()
    white_sq_list.append(white_sq)

white_sq = np.vstack(white_sq_list)
whiteref = white_sq.mean(axis=0)

white_corrected = spectral_image / whiteref

white_corrected_rgb_view = reconstruct_rgb_envi(white_corrected, envi_header)
white_corrected_rgb_view *= 1.5
white_corrected_rgb_view = white_corrected_rgb_view.clip(0, 1)

axes[1].imshow(white_corrected_rgb_view)
axes[1].set_title("After white correction")

plt.show()

# Show spectra
colors = ["r", "b", "g"]
color_names = ["red", "blue", "green"]
positions = [(320, 320), (320, 420), (320, 380)]

wavelengths = get_wavelengths(envi_header)

fig, axes = plt.subplots(2, 4, figsize=(10, 5), tight_layout=True)

axes1, axes2 = axes
for pos, color, color_name, ax in zip(positions, colors, color_names, axes2):
    ax.plot(wavelengths, white_corrected[pos[1], pos[0], :], color=color)
    ax.set_title(f"Spectra at {color_name}(corrected)")
    ax.set_xlabel("Wavelength[nm]")

for pos, color, color_name, ax in zip(positions, colors, color_names, axes1):
    ax.plot(wavelengths, spectral_image[pos[1], pos[0], :], color=color)
    ax.set_title(f"Spectra at {color_name}(original)")
    ax.set_xlabel("Wavelength[nm]")

axes1[-1].plot(wavelengths, whiteref, color="k")
axes1[-1].set_title("White reference")
axes1[-1].set_xlabel("Wavelength[nm]")

canvas = draw_multi_crosss(white_corrected_rgb_view, positions)
axes2[-1].imshow(canvas)
axes2[-1].set_title("RGB view")

plt.show()
