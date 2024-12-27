from matplotlib import pyplot as plt

from asi import path_config
from asi.draw import draw_multi_crosss, reconstruct_rgb_envi, select_area
from asi.io.load_envi import load_spectral_image
from asi.utils import get_wavelengths

specim_iq_root = path_config.measurements / "Session1" / "SpecimIQ"


# White correction with small reference
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

image_path = specim_iq_root / "404" / "capture" / "404"
spectral_image, envi_header = load_spectral_image(image_path)


original_rgb = reconstruct_rgb_envi(spectral_image, envi_header)

# original_rgb *= 1.5
# original_rgb = original_rgb.clip(0, 1)


axes[0].imshow(original_rgb)
axes[0].set_title("Before white correction")

white_path = specim_iq_root / "405" / "capture" / "405"
white_image, white_envi_header = load_spectral_image(white_path)

white_pos = (slice(200, 480), slice(100, 380))
white_rgb_view = reconstruct_rgb_envi(white_image, white_envi_header)
white_rgb_view = select_area(white_rgb_view, white_pos)

axes[1].imshow(white_rgb_view)
axes[1].set_title("Selected white area")

white_sq = white_image[white_pos]

# White correction with selected area
whiteref = white_sq.mean(axis=(0,1))

white_corrected = spectral_image / whiteref

white_corrected_rgb_view = reconstruct_rgb_envi(white_corrected, envi_header)
# white_corrected_rgb_view *= 2.5
# white_corrected_rgb_view = white_corrected_rgb_view.clip(0, 1)


axes[2].imshow(white_corrected_rgb_view)
axes[2].set_title("After white correction")

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
