import matplotlib.pyplot as plt

from asi import path_config
from asi.draw import draw_multi_crosss, reconstruct_rgb_envi
from asi.io.load_envi import load_spectral_image
from asi.preprocess import load_white_corrected
from asi.utils import get_wavelengths

session1_root = path_config.measurements / "Session1"
spec_path = session1_root / "Specim scanner/Color_checker_8_binning/capture"

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

colorchecker_path = spec_path / "solutions_scan_0110"
whiteref_path = spec_path / "WHITEREF_solutions_scan_0110"
darkref_path = spec_path / "DARKREF_solutions_scan_0110"

spectral_image, envi_header = load_spectral_image(colorchecker_path)
rgb_view = reconstruct_rgb_envi(spectral_image, envi_header)
axes[0].imshow(rgb_view)
axes[0].set_title("Before white correction")

white_corrected, envi_header = load_white_corrected(
    colorchecker_path,
    whiteref_path,
    darkref_path,
)
white_corrected_rgb_view = reconstruct_rgb_envi(white_corrected, envi_header)

white_image, _ = load_spectral_image(whiteref_path)
whiteref = white_image.mean(axis=(0, 1))

axes[1].imshow(white_corrected_rgb_view)
axes[1].set_title("After white correction")
plt.show()

# Show spectra
colors = ["r", "g", "b"]
color_names = ["red", "green", "blue"]
positions = [(300, 500), (300, 750), (600, 750)]

fig, axes = plt.subplots(2, 4, figsize=(10, 5), tight_layout=True)
wavelengths = get_wavelengths(envi_header)
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
