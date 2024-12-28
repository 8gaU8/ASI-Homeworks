import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.draw import reconstruct_gray_view, reconstruct_rgb_envi
from asi.preprocess import load_white_corrected
from asi.utils import get_wavelengths
from segmentation import plot_segmentation_results

# Load spectral image and apply white correction
session2 = path_config.measurements / "session2"
specim_scanner = session2 / "Specim scanner" / "LeavesNotesIR" / "capture"

name = "IR_scan_0464"
image_path = specim_scanner / name
darkref_path = specim_scanner / f"DARKREF_{name}"
whiteref_path = specim_scanner / f"WHITEREF_{name}"

spectral_image, envi_header = load_white_corrected(
    image_path,
    whiteref_path,
    darkref_path,
)
spectral_image = spectral_image.astype(np.float16)

rgb_view = reconstruct_gray_view(spectral_image)
rgb_view = rgb_view.clip(0, 1)
wavelengths = get_wavelengths(envi_header)


fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)


select_pos = (slice(20, 40), slice(150, 160))
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.12,
)

select_pos = (slice(55, 68), slice(105, 120))
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.12,
)

select_pos = (slice(130, 138), slice(50, 60))
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.11
)

fig.show()
fig.savefig("./fig/task2/specium-scanner-ir.png")
