from pathlib import Path

import numpy as np

from asi import path_config
from asi.io.load_nuance import load_nuance_image

session1 = path_config.measurements / "session1"
nuance = session1 / "Nuance"

root = nuance / "colorchecker 2lights"
spectral_image, wavelengths = load_nuance_image(root)

lines, samples, bands = spectral_image.shape
print(spectral_image.shape)
spectral_image_uint16 = (spectral_image).astype(np.uint16)
bil_format = spectral_image_uint16.transpose(0, 2, 1).flatten()

bil_format.tofile("saveddata/nuance.raw")

reversed_wavelengths = wavelengths[::-1]
wavelengths_hdr = ",\n\t".join(map(str, reversed_wavelengths))
wavelengths_hdr = f"wavelength = {{\n\t{wavelengths_hdr}\n}}"

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
{wavelengths_hdr}
"""

hdr_dst_path = Path("saveddata/nuance.hdr")
hdr_dst_path.write_text(header_content, encoding="utf-8")
