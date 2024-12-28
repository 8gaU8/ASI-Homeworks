import matplotlib.pyplot as plt
import numpy as np

from asi import path_config
from asi.io.load_tunable import load_tunable_image
from segmentation import plot_segmentation_results

# Configurations
WHITE_POS = (slice(600, 700), slice(230, 300))
# LOAD IMAGE
session1 = path_config.measurements / "session2"
tunable_root = session1 / "Tunable" / "green materials"

spectral_image, channels = load_tunable_image(
    tunable_root,
    name="colorchecker",
    white_pos=WHITE_POS,
)
spectral_image = spectral_image.astype(np.float64)

# Apply white correction
white_sq = spectral_image[WHITE_POS]
whiteref = white_sq.mean(axis=(0, 1))
white_corrected = spectral_image / whiteref
spectral_image /= spectral_image.max()


# Make RGB view
SELECT_CHANNELS = [7, 5, 1]
# Make RGB view
rgb_view = spectral_image[..., SELECT_CHANNELS]
# Postprocess for preview
rgb_view /= rgb_view.max()
rgb_view *= 0.8
rgb_view = rgb_view.clip(0, 1)

wavelengths = channels

fig, axes = plt.subplots(3, 3, tight_layout=True, figsize=(15, 10), dpi=80)


select_pos = (slice(0, 150), slice(1000, None))
plot_segmentation_results(
    axes[0],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.1,
)

select_pos = (slice(220, 280), slice(390, 400))
plot_segmentation_results(
    axes[1],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.05,
)

select_pos = (slice(220, 280), slice(325, 330))
plot_segmentation_results(
    axes[2],
    spectral_image,
    wavelengths,
    select_pos,
    rgb_view,
    threshold=0.11,
)

fig.show()
fig.savefig("./fig/task1/tunable.png")
