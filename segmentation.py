import matplotlib.pyplot as plt
import numpy as np

from asi.draw import reconstruct_rgb_envi
from asi.utils import get_wavelengths


def select_area(rgb_view: np.ndarray, select_pos: tuple[slice, slice]) -> np.ndarray:
    _rgb_view = rgb_view.copy()
    _rgb_view[select_pos] = [1, 0, 0]
    return _rgb_view


def get_selected_spectra(spectral_image: np.ndarray, select_pos: tuple[slice, slice]) -> np.ndarray:
    selected_cube = spectral_image[select_pos]
    selected_spectra = selected_cube.mean(axis=(0, 1))
    return selected_spectra


def plot_selected_spectra(
    spectral_image: np.ndarray, select_pos: tuple[slice, slice], wavelengths, ax
) -> None:
    selected_cube = spectral_image[select_pos]

    spectra_per_line = selected_cube.mean(axis=0)
    for line in spectra_per_line[:10]:
        ax.plot(wavelengths, line, alpha=0.2)
    selected_spectra = selected_cube.mean(axis=(0, 1))
    ax.plot(wavelengths, selected_spectra, label="Mean", color="black")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_title("Spectra of selected region")
    ax.legend()


def get_segmented_image(
    spectral_image: np.ndarray, selected_spectra: np.ndarray, threshold: float
) -> np.ndarray:
    diff = np.abs(spectral_image - selected_spectra)
    mask = diff < threshold
    mask = mask.all(axis=2)
    return mask


def get_segmented_rgb_view(rgb_view: np.ndarray, mask: np.ndarray) -> np.ndarray:
    segmented_rgb_view = rgb_view.copy()
    segmented_rgb_view[mask] = [1, 0, 0]
    gray = rgb_view.mean(axis=2)
    segmented_rgb_view[~mask] = gray[~mask][:, None]
    segmented_rgb_view *= 1.5
    segmented_rgb_view = segmented_rgb_view.clip(0, 1)
    return segmented_rgb_view


def reconstruct_gray_rgb(spectral_image):
    channels = spectral_image.shape[2]
    ch_idx = channels // 2

    gray_view = spectral_image[..., ch_idx].copy()
    gray_view = gray_view.clip(0, 1)
    gray_view = np.stack((gray_view, gray_view, gray_view), axis=-1)
    gray_view = gray_view.astype(np.float32)

    return gray_view


def plot_segmentation_results(
    spectral_image, envi_header, select_pos, threshold=0.2, use_gray=False
):
    if not use_gray:
        rgb_view = reconstruct_rgb_envi(spectral_image, envi_header)
    else:
        rgb_view = reconstruct_gray_rgb(spectral_image)

    # calulate all we need
    selected_spectra = get_selected_spectra(spectral_image, select_pos)
    mask = get_segmented_image(spectral_image, selected_spectra, threshold)

    # prepare plot
    fig, axes = plt.subplots(1, 3, tight_layout=True, figsize=(15, 5))
    org_ax = axes[0]
    spectra_ax = axes[1]
    segmented_ax = axes[2]

    # plot selected region
    masked_rgb = select_area(rgb_view, select_pos)
    masked_rgb *= 1.5
    masked_rgb = masked_rgb.clip(0, 1)
    org_ax.imshow(masked_rgb)
    org_ax.set_title("Selected region")

    # plot selected spectra
    wavelengths = get_wavelengths(envi_header)
    plot_selected_spectra(spectral_image, select_pos, wavelengths, spectra_ax)

    # plot segmented
    segmented_rgb_view = get_segmented_rgb_view(rgb_view, mask)
    segmented_ax.imshow(segmented_rgb_view)
    segmented_ax.set_title("Segmented region")
    return fig
