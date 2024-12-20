from pathlib import Path

import numpy as np
import tifffile as tiff
from natsort import natsorted


def load_nuance_image(tiff_root: Path) -> tuple[np.ndarray, list[float]]:
    wavelengths: list[float] = []
    imgs = []
    tiff_list = list(tiff_root.glob("*.tif"))
    tiff_list = natsorted(tiff_list, reverse=False)
    for tiff_path in tiff_list:
        img = tiff.imread(tiff_path)
        wavelength = float(tiff_path.stem.split("_")[-1])
        imgs.append(img)
        wavelengths.append(wavelength)

    spectral_image = np.stack(imgs, axis=-1)
    spectral_image = spectral_image.astype(np.float64)
    return spectral_image, wavelengths
