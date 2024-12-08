import numpy as np
from .envi import parse_envi_header


def load_spectral_image(file_stem: str) -> tuple[np.ndarray, dict[str, str]]:
    """Loads spectral image from ENVI format."""
    # Load ENVI header
    hdr_file = file_stem + ".hdr"
    with open(hdr_file, "r") as f:
        lines = f.readlines()
    envi_header = parse_envi_header(lines)

    # Load parameters
    hdr_lines = int(envi_header["lines"])
    hdr_samples = int(envi_header["samples"])
    hdr_bands = int(envi_header["bands"])

    # Load raw data
    raw_file = file_stem + ".raw"
    with open(raw_file, "rb") as f:
        raw = np.fromfile(f, dtype=np.uint16)

    # Reshape 1D to 3D. The order 'lines, bands, samples' for interleave = BIL case
    new_shape: tuple[int, int, int] = (hdr_lines, hdr_bands, hdr_samples)
    spectral_image = raw.reshape(new_shape)

    return spectral_image, envi_header
