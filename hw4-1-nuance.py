import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.draw import reconstruct_rgb
from asi.io.load_nuance import load_nuance_image
from segmentation import plot_segmentation_results

# Load spectral image and apply white correction
session2 = path_config.measurements / "session2"
nuance = session2 / "Nuance"
root = nuance / "greenmaterials"

# load spectral image
spectral_image, wavelengths = load_nuance_image(root)
spectral_image = spectral_image.astype(np.float64)

white_pos = (slice(320, 400), slice(100, 200))

# White correction with selected area
white_sq = spectral_image[white_pos]

# replace nonzero elements with minimum value
nonzero_elements = white_sq[white_sq != 0]
min_elm = nonzero_elements.min()
white_sq = white_sq.clip(min_elm, None)

# apply white correction
whiteref = white_sq.mean(axis=(0, 1))
spectral_image /= whiteref

rgb_view = reconstruct_rgb(spectral_image, wavelengths)


fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)

select_pos = (slice(600, 800), slice(0, 200))
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.11,
)

select_pos = (slice(915, 927), slice(1212, 1235))
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.2,
)
select_pos = (slice(580, 640), slice(500, 520))
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.080,
)

fig.show()
fig.savefig("./fig/task1/nuance.png")
