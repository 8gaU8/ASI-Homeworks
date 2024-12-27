import numpy as np
from matplotlib import pyplot as plt

from asi import path_config
from asi.draw import draw_multi_crosss, reconstruct_rgb_envi, select_area
from asi.io.load_envi import load_spectral_image
from asi.utils import get_wavelengths

specim_iq_root = path_config.measurements / "session1" / "SpecimIQ"

path_404 = specim_iq_root / "404" / "capture"
path_405 = specim_iq_root / "405" / "capture"


# White correction with small reference
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

image_path = path_404 / "404"
spectral_image, envi_header = load_spectral_image(image_path)


white_pos = slice(65, 102), slice(332, 370)

original_rgb = reconstruct_rgb_envi(spectral_image, envi_header)
original_rgb *= 2.5
original_rgb = original_rgb.clip(0, 1)
original_rgb = select_area(original_rgb, white_pos)

axes[0].imshow(original_rgb)
axes[0].set_title("Before white correction")


white_sq = spectral_image[white_pos]
# replace nonzero elements with minimum value
nonzero_elements = white_sq[white_sq != 0]
min_elm = nonzero_elements.min()
white_sq = white_sq.clip(min_elm, None)

# apply white correction
whiteref = white_sq.mean((0, 1))
white_corrected = spectral_image / whiteref
white_corrected /= white_corrected.max()
white_corrected = white_corrected.clip(0, 1)

white_corrected_rgb_view = reconstruct_rgb_envi(white_corrected, envi_header)
white_corrected_rgb_view *= 2.5
white_corrected_rgb_view = white_corrected_rgb_view.clip(0, 1)


axes[1].imshow(white_corrected_rgb_view)
axes[1].set_title("After white correction")

plt.show()

# Show spectra
wavelengths = get_wavelengths(envi_header)
colors = ["r", "g"]
positions = [(320, 220), (320, 420)]
canvas = draw_multi_crosss(white_corrected_rgb_view, positions)

plt.rcParams["figure.dpi"] = 100
fig, axes = plt.subplots(1, 4, figsize=(10, 5), tight_layout=True)

for pos, color, ax in zip(positions, colors, axes[:2]):
    ax.plot(wavelengths, spectral_image[pos[1], pos[0], :], color=color)
    ax.plot(wavelengths, white_corrected[pos[1], pos[0], :], color=color)
    ax.set_title(f"Spectrum at {pos[0]}th pixel")
    ax.set_xlabel("Wavelength[nm]")

axes[2].plot(wavelengths, whiteref, color="b")
axes[2].set_title("White reference spectrum")

axes[3].imshow(canvas)
axes[3].set_title("RGB view of spectral image")

plt.show()
