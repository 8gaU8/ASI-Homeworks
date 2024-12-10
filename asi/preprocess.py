# Author: Yuya HAGA

import numpy as np


def white_correction(
    spectral_image: np.ndarray,
    whiteref: np.ndarray,
    darkref: np.ndarray,
) -> np.ndarray:
    white_corrected = (spectral_image - darkref) / (whiteref - darkref)
    return white_corrected
