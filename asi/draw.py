# Author: Yuya HAGA


import cv2
import numpy as np

from .utils import search_closest_index


def reconstruct_rgb(
    spectral_image: np.ndarray,
    wavelengths: list[float],
    rgb_wavelengths: tuple[float, float, float] = (632.15, 528.03, 443.56),
):
    """Reconstructs RGB image from spectral image with given three wavelengths."""
    # Get wavelengths from header
    rgb_indeces = [search_closest_index(wavelengths, wl) for wl in rgb_wavelengths]

    # Prepare RGB view placeholder
    lines, samples, _bands = spectral_image.shape
    rgb_view = np.empty((lines, samples, 3))
    # Reconstruct RGB image
    for idx, ch in enumerate(rgb_indeces):
        rgb_view[:, :, idx] = spectral_image[:, :, ch] / np.amax(
            spectral_image[:, :, ch]
        )
    return rgb_view


def reconstruct_rgb_envi(
    spectral_image: np.ndarray,
    envi_header: dict[str, str],
    rgb_wavelengths: tuple[float, float, float] = (632.15, 528.03, 443.56),
):
    # """Reconstructs RGB image from spectral image with given three wavelengths."""
    # # Get wavelengths from header
    wavelengths = [float(data) for data in envi_header["wavelength"].split(",")]
    rgb_indeces = [search_closest_index(wavelengths, wl) for wl in rgb_wavelengths]
    lines, samples, _bands = spectral_image.shape
    rgb_view = np.empty((lines, samples, 3))
    # Reconstruct RGB image
    for idx, ch in enumerate(rgb_indeces):
        rgb_view[:, :, idx] = spectral_image[:, :, ch] / np.amax(
            spectral_image[:, :, ch]
        )
    return rgb_view


def draw_cross(rgb: np.ndarray, position: tuple[int, int]):
    """Draws cross on RGB view at given position"""
    img = rgb.copy()
    img = cv2.drawMarker(
        img,
        position,
        color=(1, 1, 1),
        markerType=cv2.MARKER_CROSS,
        markerSize=50,
        thickness=10,
        line_type=cv2.LINE_8,
    )
    img = cv2.drawMarker(
        img,
        position,
        color=(0, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=45,
        thickness=5,
        line_type=cv2.LINE_4,
    )
    return img


def draw_multi_crosss(img: np.ndarray, positions: list[tuple[int, int]]):
    """Draws multiple cross on RGB view at given positions"""

    for position in positions:
        img = draw_cross(img, position)
    return img


def select_area(rgb_view: np.ndarray, select_pos: tuple[slice, slice]) -> np.ndarray:
    _rgb_view = rgb_view.copy()
    _rgb_view[select_pos] = [1, 0, 0]
    return _rgb_view


def reconstruct_gray_view(spectral_image):
    channels = spectral_image.shape[2]
    ch_idx = channels // 2

    gray_view = spectral_image[..., ch_idx].copy()
    gray_view = gray_view.clip(0, 1)
    gray_view = np.stack((gray_view, gray_view, gray_view), axis=-1)
    gray_view = gray_view.astype(np.float32)

    return gray_view
