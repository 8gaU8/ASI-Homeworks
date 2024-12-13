from pathlib import Path

_root = Path(
    "/Users/hagayuya/Library/CloudStorage/GoogleDrive-haga.yuya2.2016@gmail.com/マイドライブ/",
)

measurements = _root / "ASI course 2024/group3"
camera_from_japan = _root / "Camera from Japan"
lectures = _root / "Lectures+Exercises"
image_of_coin = _root / "ASI/Image of coin (Senop camera)"

# check if the all paths exist, if not, raise FileNotFoundError
for path in [_root, measurements, camera_from_japan, lectures, image_of_coin]:
    if not path.exists():
        msg = f"{path} does not exist"
        raise FileNotFoundError(msg)
