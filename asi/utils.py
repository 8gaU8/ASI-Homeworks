import numpy as np


def search_closest_index(wavelengths: list[float], target_wavelength: float) -> int:
    """Finds the index of the closest wavelength to the target wavelength from value of envi header."""
    closest_index = int(np.argmin(np.abs(np.array(wavelengths) - target_wavelength)))
    return closest_index


def get_wavelengths(envi_header: dict[str, str]) -> list[float]:
    """Returns wavelengths from envi header."""
    wavelength = [float(wl) for wl in envi_header["wavelength"].split(",")]
    return wavelength
