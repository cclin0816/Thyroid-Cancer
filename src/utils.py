import math
import cv2
import numpy
import openslide
import json
import sys
import xxhash
from IPython.display import display
from pathlib import Path
from typing import Any
from PIL import Image


def load_json(path: Path) -> Any:
    with path.open("r") as fp:
        return json.load(fp)


def dump_json(path: Path, obj: Any) -> None:
    with path.open("w") as fp:
        json.dump(obj, fp)


# default NOTICE
cur_log_lvl = 3
log_lvl_lut = {"DEBUG": 5, "INFO": 4, "NOTICE": 3, "WARN": 2, "ERR": 1, "CRIT": 0}


def log(lvl: str, msg: str) -> None:
    global log_lvl_lut
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl is None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    if log_lvl <= cur_log_lvl:
        print(f"[{lvl}] {msg}", file=sys.stderr)


def set_log_lvl(lvl: str) -> None:
    global log_lvl_lut
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl is None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    cur_log_lvl = log_lvl


Bbox = tuple[int, int, int, int]
""" type alias of (x_min, y_min, width, height) """


def get_bbox(data) -> Bbox:
    return (data["x_min"], data["y_min"], data["width"], data["height"])  # type: ignore


def resize_keep_ratio(image: Image.Image, width: int, height: int) -> Image.Image:
    # image.thumbnail() fails when image is smaller than width, height
    ratio = image.width / image.height
    if ratio > 1.0:
        w = width
        h = round(w / ratio)
    else:
        h = height
        w = round(h * ratio)

    return image.resize((w, h), Image.Resampling.LANCZOS)


def grid_composite(
    images: list[Image.Image], row_len: int = 16, width: int = 100, height: int = 100
) -> Image.Image:
    col_len = math.ceil(len(images) / row_len)
    grid = Image.new("RGB", (width * row_len, height * col_len))

    for idx, image in enumerate(images):
        img = resize_keep_ratio(image, width, height)
        x = (idx % row_len) * width + (width - img.width) // 2
        y = (idx // row_len) * height + (height - img.height) // 2
        grid.paste(img, (x, y))

    return grid


def crop_rotated_rectangle(
    image: numpy.ndarray,
    center: tuple[float, float],
    size: tuple[int, int],
    angle: float,
) -> numpy.ndarray:
    # crop small square first to speed up rotate
    (w, h) = size
    cs = math.ceil(math.sqrt((w / 2) ** 2 + (h / 2) ** 2) * 2)
    cc = (cs - 1) / 2
    cs = (cs, cs)
    cc = (cc, cc)

    crop = cv2.getRectSubPix(image, cs, center)
    rot_mat = cv2.getRotationMatrix2D(cc, angle, 1)
    rotate = cv2.warpAffine(crop, rot_mat, cs)
    return cv2.getRectSubPix(rotate, size, cc)


class SlideReader:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.cache: dict[str, openslide.OpenSlide | openslide.ImageSlide] = {}

    def read_bbox(self, slide: str, bbox: Bbox) -> Image.Image:
        handle = self.load(slide)
        return handle.read_region((bbox[0], bbox[1]), 0, (bbox[2], bbox[3]))

    def read(self, slide: str, level: int = 0) -> Image.Image:
        handle = self.load(slide)
        return handle.read_region((0, 0), level, handle.level_dimensions[level])

    def load(self, slide: str) -> openslide.OpenSlide | openslide.ImageSlide:
        handle = self.cache.get(slide)
        if handle is None:
            handle = openslide.open_slide(self.folder / (slide + ".ndpi"))
            # purge cache, doing smth like LRU is not really beneficial
            if len(self.cache) > 16:
                self.cache.clear()
            self.cache[slide] = handle
        return handle


class BufferedDisplay:
    def __init__(
        self, row_len: int = 16, col_len: int = 1, width: int = 100, height: int = 100
    ) -> None:
        self.buf = []
        self.row_len = row_len
        self.col_len = col_len
        self.buf_len = row_len * col_len
        self.width = width
        self.height = height

    def flush(self) -> None:
        if len(self.buf) > 0:
            grid = grid_composite(self.buf, self.row_len, self.width, self.height)
            display(grid)
            self.buf.clear()

    def display(self, image: Image.Image) -> None:
        self.buf.append(image)
        if len(self.buf) >= self.buf_len:
            self.flush()


def mass_hash(slide: str, bbox: Bbox) -> int:
    return xxhash.xxh3_64_intdigest(f"{slide},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
