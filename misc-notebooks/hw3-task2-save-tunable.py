from pathlib import Path

import numpy as np

from asi import path_config
from asi.io.load_tunable import load_tunable_image

session1 = path_config.measurements / "session1"
tunable_root = session1 / "Tunable light sorces" / "ImagesASI"

WHITE_POS = (slice(510, 550), slice(230, 260))
spectral_image, channels = load_tunable_image(
    tunable_root,
    name="colorchecker",
    white_pos=WHITE_POS,
)
spectral_image = spectral_image.astype(np.float64)


lines, samples, bands = spectral_image.shape
spectral_image_uint16 = (spectral_image).astype(np.uint16)
bil_format = spectral_image_uint16.transpose(0, 2, 1).flatten()

bil_format.tofile("saveddata/tunable.raw")


header_content = f"""ENVI
ENVI description = {{File Imported into ENVI}}
file type = ENVI
lines = {lines}
samples = {samples}
bands = {bands}
interleave = bil
data type = 12
header offset = 0
byte order = 0
"""

dst_hdr_path = Path("saveddata/tunable.hdr")
dst_hdr_path.write_text(header_content, encoding="utf-8")
