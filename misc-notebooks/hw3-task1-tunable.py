import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.draw import draw_multi_crosss, select_area
from asi.io import load_tunable_image

# Configurations
WHITE_POS = (slice(510, 550), slice(230, 260))
SELECT_CHANNELS = [9, 4, 0]
# LOAD IMAGE
session1 = path_config.measurements / "session1"
tunable_root = session1 / "Tunable light sorces" / "ImagesASI"

spectral_image, channels = load_tunable_image(
    tunable_root, name="colorchecker", white_pos=WHITE_POS
)
spectral_image = spectral_image.astype(np.float64)

# Make RGB view
rgb_view = spectral_image[..., SELECT_CHANNELS]
# Postprocess for preview
rgb_view /= rgb_view.max()
rgb_view *= 0.8
rgb_view = rgb_view.clip(0, 1)

# Select white area for correction
rgb_view = select_area(rgb_view, WHITE_POS)

# Apply white correction
white_sq = spectral_image[WHITE_POS]
whiteref = white_sq.mean(axis=(0, 1))
white_corrected = spectral_image / whiteref

# RGB view after white correction
white_corrected_rgb_view = white_corrected[..., SELECT_CHANNELS]
white_corrected_rgb_view /= white_corrected_rgb_view.max()
white_corrected_rgb_view *= 1.2
white_corrected_rgb_view = white_corrected_rgb_view.clip(0, 1)


# Plot images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb_view)
axes[0].set_title("Before white correction")

axes[1].imshow(white_corrected_rgb_view)
axes[1].set_title("After white correction")

plt.show()

# show spectra
red_pos = (450, 450)
blue_pos = (400, 280)
green_pos = (400, 400)


pos_list = (red_pos, blue_pos, green_pos)
colors = ["r", "b", "g"]
color_names = ["red", "blue", "green"]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

org_axes, white_axes = axes
for axes, image in zip([org_axes, white_axes], [spectral_image, white_corrected]):
    for pos, color_name, color, ax in zip(pos_list, color_names, colors, axes):
        ax.plot(channels, image[pos[1], pos[0], :], color=color)
        ax.set_title(f"{color_name} spectra")


org_axes[-1].plot(whiteref, color="k")
org_axes[-1].set_title("White reference")
org_axes[-1].set_xlabel("Wavelength[nm]")

canvas = draw_multi_crosss(white_corrected_rgb_view, pos_list)
white_axes[-1].imshow(canvas)
white_axes[-1].set_title("RGB view")

plt.show()
