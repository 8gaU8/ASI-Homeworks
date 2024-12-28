import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.draw import reconstruct_gray_view
from asi.preprocess import load_white_corrected
from asi.utils import get_wavelengths
from segmentation import plot_segmentation_results

session2 = path_config.measurements / "session2"

specim_iq = session2 / "Specim scanner" / "PowdersIR" / "capture"

image_path = specim_iq / "IR_scan_0462"
darkref_path = specim_iq / "DARKREF_IR_scan_0462"
whiteref_path = specim_iq / "WHITEREF_IR_scan_0462"

spectral_image, envi_header = load_white_corrected(
    image_path, whiteref_path, darkref_path
)
spectral_image = spectral_image.astype(np.float16)

rgb_view = reconstruct_gray_view(spectral_image)
wavelengths = get_wavelengths(envi_header)


fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)


threshold = 0.15
select_pos = slice(70, 90), slice(80, 90)
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold,
)
select_pos = slice(42, 52), slice(80, 90)
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold,
)

select_pos = slice(135, 145), slice(135, 145)
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold,
)

fig.show()
fig.savefig("./fig/task3/specium-scanner-ir.png")
