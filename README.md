# Python module `asi` for Advanced Spectral Imaging cource


A Python module for advanced spectral imaging including parsing ENVI header file, load spectral image like FreeLook, and make RGB preview of spectral image.

## Structure

- `/asi.io`
  - Scripts to load various spctral image formats, such as ENVI, Nuance Camera, and Tunable Light Source Camera
- `/asi.path_config`
  - Path configuration file
  - TODO: this file should be exported from the repository
- `/asi.preprocess`
  - White and dark correction
- `/segmentation.py`
  - Some scipts for segment by spectra.
  - TODO: this file should be included in `asi.postprocess` or `asi.segmentation`

## Usage

- import module
  - `import asi`

- Load spectral image and show RGB

```python
import matplotlib.pyplot as plt
import asi
path_to_spectral_image = "ASI course 2024/group3/Session1/Specim scanner/Color_checker_8_binning/capture/solutions_scan_0110"

spectral_image, envi_header = asi.load_spectral_image(path_to_spectral_image)
rgb_view = asi.reconstruct_rgb(spectral_image, envi_header)
plt.imshow(rgb_view)

```

- Load and apply white correction

```python
import asi
spectral_image_path = "ASI course 2024/group3/Session1/Specim scanner/Color_checker_8_binning/capture/solutions_scan_0110"
whiteref_path = "ASI course 2024/group3/Session1/Specim scanner/Color_checker_8_binning/capture/WHITEREF_solutions_scan_0110"
darkref_path = "ASI course 2024/group3/Session1/Specim scanner/Color_checker_8_binning/capture/DARKREF_solutions_scan_0110"

spectral_image, envi_hdr = asi.load_spectral_image(spectral_image_path)
whiteref, _wh_hdr = asi.load_spectral_image(whiteref_path)
darkref, _dr_hdr = asi.load_spectral_image(darkref_path)

# White correction
white_corrected = (spectral_image - darkref) / (whiteref - darkref)
rgb_view = asi.reconstruct_rgb(white_corrected, envi_header)
plt.imshow(rgb_view)
```