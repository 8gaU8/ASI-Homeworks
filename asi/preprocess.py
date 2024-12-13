# Author: Yuya HAGA

from pathlib import Path

import numpy as np

from .io import load_spectral_image


def white_correction(
    spectral_image: np.ndarray,
    whiteref: np.ndarray,
    darkref: np.ndarray,
) -> np.ndarray:
    white_corrected = (spectral_image - darkref) / (whiteref - darkref)
    return white_corrected


def load_white_corrected(
    file_path: Path,
    whiteref_path: Path,
    darkref_path: Path,
):
    spectral_image, envi_header = load_spectral_image(file_path)
    whiteref, _ = load_spectral_image(whiteref_path)
    darkref, _ = load_spectral_image(darkref_path)
    whiteref = whiteref.mean(axis=0)
    darkref = darkref.mean(axis=0)
    white_corrected = white_correction(spectral_image, whiteref, darkref)
    return white_corrected, envi_header


def white_correction_sq(
    spectral_image: np.ndarray,
    white_pos: tuple[slice, slice],
) -> np.ndarray:
    white_sq = spectral_image[white_pos]
    whiteref = white_sq.mean((0, 1))
    spectral_image = spectral_image / whiteref
    return spectral_image
