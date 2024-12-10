import numpy as np  # noqa: A005

from .envi import parse_envi_header


def load_spectral_image(file_stem: str) -> tuple[np.ndarray, dict[str, str]]:
    """Loads spectral image from ENVI format."""
    # Load ENVI header
    hdr_file = file_stem + ".hdr"
    with open(hdr_file, encoding="utf-8") as f:
        header_content = f.readlines()
    envi_header = parse_envi_header(header_content)

    # Load parameters
    interleave = str(envi_header["interleave"])
    lines = int(envi_header["lines"])
    samples = int(envi_header["samples"])
    bands = int(envi_header["bands"])

    # Load raw data
    raw_file = file_stem + ".raw"
    with open(raw_file, "rb") as f:
        raw = np.fromfile(f, dtype=np.uint16)

    # Reshape 1D to 3D. The order 'lines, bands, samples' for interleave = BIL case
    if interleave.upper() == "BIL":
        new_shape = (lines, bands, samples)
    elif interleave.upper() == "BIP":
        new_shape = (lines, samples, bands)
    elif interleave.upper() == "BSQ":
        new_shape = (bands, samples, lines)
    else:
        msg = f"Interleave {interleave} not supported."
        raise ValueError(msg)

    spectral_image = raw.reshape(new_shape)

    return spectral_image, envi_header
