import cv2
import numpy as np

from .utils import search_closest_index


def reconstruct_rgb(
    spectral_image: np.ndarray,
    envi_header: dict[str, str],
    rgb_wavelengths: tuple[float, float, float] = (632.15, 528.03, 443.56),
):
    """Reconstructs RGB image from spectral image with given three wavelengths."""
    # Get wavelengths from header
    wavelengths = [float(data) for data in envi_header["wavelength"].split(",")]
    rgb_indeces = [search_closest_index(wavelengths, wl) for wl in rgb_wavelengths]

    # Prepare RGB view placeholder
    lines, samples, _bands = spectral_image.shape
    rgb_view = np.empty((lines, samples, 3))
    # Reconstruct RGB image
    for idx, ch in enumerate(rgb_indeces):
        rgb_view[:, :, idx] = spectral_image[:, :, ch] / np.amax(spectral_image[:, :, ch])
    return rgb_view


def draw_cross(rgb_view: np.ndarray, position: tuple[int, int]):
    """Draws cross on RGB view at given position"""
    img = rgb_view.copy()
    img = cv2.drawMarker(
        img,
        position,
        color=(1, 1, 1),
        markerType=cv2.MARKER_CROSS,
        markerSize=50,
        thickness=10,
        line_type=cv2.LINE_8,
    )
    return img


def draw_multi_crosss(rgb_view: np.ndarray, positions: list[tuple[int, int]]):
    """Draws multiple cross on RGB view at given positions"""
    img = rgb_view.copy()
    for position in positions:
        img = cv2.drawMarker(
            img,
            position,
            color=(1, 1, 1),
            markerType=cv2.MARKER_CROSS,
            markerSize=50,
            thickness=10,
            line_type=cv2.LINE_8,
        )
    return img
