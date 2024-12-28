import numpy as np
from matplotlib import pyplot as plt

from asi import path_config
from asi.draw import draw_multi_crosss, reconstruct_rgb_envi, select_area
from asi.io.load_envi import load_spectral_image
from asi.preprocess import load_white_corrected
from asi.utils import get_wavelengths
from segmentation import plot_segmentation_results

name = "415"
specim_iq_root = path_config.measurements / "Session2" / "SpecimIQ"

path_404 = specim_iq_root / name / "capture"

# White correction with small reference

image_path = path_404 / name
whiteref_path = path_404 / f"WHITEREF_{name}"
darkref_path = path_404 / f"DARKREF_{name}"

spectral_image, envi_header = load_white_corrected(
    image_path, whiteref_path, darkref_path
)

# apply white correction
white_pos = (slice(430, None), slice(200, 280))
white_area = spectral_image[white_pos]
whiteref = white_area.mean(axis=(0, 1))
spectral_image /= whiteref

# Make RGB view
rgb_view = reconstruct_rgb_envi(spectral_image, envi_header)
rgb_view /= rgb_view.max()
rgb_view *= 3
rgb_view = rgb_view.clip(0, 1)


wavelengths = get_wavelengths(envi_header)


fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)


select_pos = (slice(15, 20), slice(75, 90))
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.2,
)

select_pos = (slice(330, 340), slice(145, 155))
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.08,
)

select_pos = (slice(130, 138), slice(50, 60))
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.11,
)

fig.show()
fig.savefig("./fig/task2/specim-iq.png")
