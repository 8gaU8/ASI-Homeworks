from pathlib import Path

import cv2
import numpy as np

# CONSTS
MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0


def parse_png_path(path: Path) -> tuple[str, int, float]:
    # parse path of tunable files
    path_parts = path.stem.split(",")
    name = path_parts[0]
    ch_info = path_parts[1]
    exp_info = path_parts[2]
    ch_id = int(ch_info.strip().split(" ")[1])
    exp_id = float(exp_info.strip().split(" ")[1])
    return name, ch_id, exp_id


def gen_template(ch: int) -> str:
    return f"*ch {ch},*.png"


def get_score(im: np.ndarray) -> int:
    nb_max = (im == MAX_PIXEL_VALUE).sum()
    nb_min = (im == MIN_PIXEL_VALUE).sum()
    return nb_max + nb_min


def load_tunable_image(
    tunable_root: Path, white_pos: tuple[slice, slice]
) -> tuple[np.ndarray, list[int]]:
    png_list = list(tunable_root.glob("*.png"))
    channels = {parse_png_path(p)[1] for p in png_list}
    channels = sorted(channels)

    imgs = []
    best_im = None
    # scoreが最小の画像を選ぶ
    best_score = 1e9
    for ch in channels:
        template = gen_template(ch)
        for png_path in tunable_root.glob(template):
            im = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if im[white_pos[0], white_pos[1]].max() == MAX_PIXEL_VALUE:
                continue
            score = get_score(im)
            if score < best_score:
                best_score = score
                best_im = im
        if best_im is None:
            msg = f"Channel {ch} not found."
            raise ValueError(msg)

        imgs.append(best_im)

    spectral_image = np.stack(imgs, axis=-1)
    return spectral_image, channels
