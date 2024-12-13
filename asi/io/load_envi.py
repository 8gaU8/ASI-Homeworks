# Author: Yuya HAGA

from pathlib import Path

import numpy as np

from .parse_envi import load_envi_header


def load_spectral_image(file_stem: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Loads spectral image from ENVI format."""
    # Load ENVI header
    hdr_file = file_stem.with_suffix(".hdr")
    envi_header = load_envi_header(hdr_file)

    # Load parameters
    interleave = str(envi_header["interleave"])
    lines = int(envi_header["lines"])
    samples = int(envi_header["samples"])
    bands = int(envi_header["bands"])
    data_type = int(envi_header.get("data type", 12))

    # Map ENVI data type to NumPy dtype
    data_type_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }

    if data_type not in data_type_map:
        msg = f"Unsupported data type: {data_type}"
        raise ValueError(msg)

    dtype = data_type_map[data_type]

    # Load raw data
    raw_file = file_stem.with_suffix(".raw")
    with open(raw_file, "rb") as f:
        raw = np.fromfile(f, dtype=dtype)

    # define shape and transpose order by interleave method
    if interleave.upper() == "BIL":
        new_shape = (lines, bands, samples)
        axis_order = (0, 2, 1)
    elif interleave.upper() == "BIP":
        new_shape = (lines, samples, bands)
        axis_order = (0, 1, 2)
    elif interleave.upper() == "BSQ":
        new_shape = (bands, samples, lines)
        axis_order = (0, 2, 1)
    else:
        msg = f"Interleave {interleave} not supported."
        raise ValueError(msg)

    spectral_image = raw.reshape(new_shape)
    # change axis order to 'lines, samples, bands'
    spectral_image = np.transpose(spectral_image, axis_order)

    return spectral_image, envi_header
