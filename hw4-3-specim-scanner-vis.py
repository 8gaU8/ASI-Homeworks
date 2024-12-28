import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.draw import reconstruct_rgb_envi
from asi.preprocess import load_white_corrected
from asi.utils import get_wavelengths
from segmentation import plot_segmentation_results

# Load spectral image and apply white correction
session2 = path_config.measurements / "session2"
specim_scanner = session2 / "Specim scanner" / "GreenSamplesVisible" / "capture"

name = "solutions_scan_0145"
image_path = specim_scanner / name
darkref_path = specim_scanner / f"DARKREF_{name}"
whiteref_path = specim_scanner / f"WHITEREF_{name}"

spectral_image, envi_header = load_white_corrected(
    image_path,
    whiteref_path,
    darkref_path,
)
spectral_image = spectral_image.astype(np.float16)

rgb_view = reconstruct_rgb_envi(spectral_image, envi_header)
rgb_view *= 3
rgb_view = rgb_view.clip(0, 1)
wavelengths = get_wavelengths(envi_header)


fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)


select_pos = (slice(1245, 1260), slice(1750, 1770))
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.3,
)
select_pos = (slice(200, 240), slice(1600, 1700))
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.3,
)
select_pos = (slice(200, 400), slice(0, 250))
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
)

fig.show()
fig.savefig("./fig/task1/specium-scanner-vis.png")
