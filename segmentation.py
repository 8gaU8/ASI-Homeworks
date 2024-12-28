import matplotlib.pyplot as plt
import numpy as np

from asi.draw import select_area


def get_selected_spectra(
    spectral_image: np.ndarray,
    select_pos: tuple[slice, slice],
) -> np.ndarray:
    selected_cube = spectral_image[select_pos]
    selected_spectra = selected_cube.mean(axis=(0, 1))
    return selected_spectra


def plot_selected_spectra(
    spectral_image: np.ndarray,
    select_pos: tuple[slice, slice],
    wavelengths: list[float] | list[int],
    ax: plt.Axes,
) -> None:
    selected_cube = spectral_image[select_pos]

    spectra_per_line = selected_cube.mean(axis=0)
    for line in spectra_per_line[:10]:
        ax.plot(wavelengths, line, alpha=0.2)
    selected_spectra = selected_cube.mean(axis=(0, 1))
    ax.plot(wavelengths, selected_spectra, label="Mean", color="black")
    if isinstance(wavelengths[0], int):
        ax.set_xlabel("Channel")
    else:
        ax.set_xlabel("Wavelength [nm]")
    ax.set_title("Spectra of selected region")
    ax.legend()


def get_segmented_image(
    spectral_image: np.ndarray,
    selected_spectra: np.ndarray,
    threshold: float,
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
    segmented_rgb_view = segmented_rgb_view.clip(0, 1)
    return segmented_rgb_view


def plot_segmentation_results(
    axes: list[plt.Axes],
    spectral_image: np.ndarray,
    wavelengths: list[float] | list[int],
    select_pos: tuple[slice, slice],
    rgb_view: np.ndarray,
    threshold=0.2,
):
    if len(axes) != 3:
        msg = "axes must be a list of 3 axes"
        raise ValueError(msg)

    # calulate all we need
    selected_spectra = get_selected_spectra(spectral_image, select_pos)
    mask = get_segmented_image(spectral_image, selected_spectra, threshold)

    # prepare plot
    org_ax = axes[0]
    spectra_ax = axes[1]
    segmented_ax = axes[2]

    # plot selected region
    masked_rgb = select_area(rgb_view, select_pos)
    masked_rgb = masked_rgb.clip(0, 1)
    org_ax.imshow(masked_rgb)
    org_ax.set_title("Selected region")

    # plot selected spectra
    plot_selected_spectra(spectral_image, select_pos, wavelengths, spectra_ax)

    # plot segmented
    segmented_rgb_view = get_segmented_rgb_view(rgb_view, mask)
    segmented_ax.imshow(segmented_rgb_view)
    segmented_ax.set_title("Segmented region")
