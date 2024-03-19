from IPython.display import display
from pandas import Series
from pathlib import Path
from typing import Any
from PIL import Image
import image_tools
import openslide
import json
import sys
import xxhash


def load_json(path: Path) -> Any:
    with path.open("r") as fp:
        return json.load(fp)


def dump_json(path: Path, obj: Any):
    with path.open("w") as fp:
        json.dump(obj, fp)


cur_log_lvl = 3
log_lvl_lut = {"DEBUG": 5, "INFO": 4, "NOTICE": 3, "WARN": 2, "ERR": 1, "CRIT": 0}


def log(lvl: str, msg: str):
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl is None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    if log_lvl <= cur_log_lvl:
        print(f"[{lvl}] {msg}", file=sys.stderr)


def set_log_lvl(lvl: str):
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl is None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    cur_log_lvl = log_lvl


Bbox = tuple[int, int, int, int]
""" type alias of (x_min, y_min, width, height) """


def get_bbox(data: Series) -> Bbox:
    return (data["x_min"], data["y_min"], data["width"], data["height"])  # type: ignore


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
            if len(self.cache) > 64:
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
            grid = image_tools.grid_composite(
                self.buf, self.row_len, self.width, self.height
            )
            display(grid)
            self.buf.clear()

    def display(self, image: Image.Image) -> None:
        self.buf.append(image)
        if len(self.buf) >= self.buf_len:
            self.flush()


def mass_hash(slide: str, bbox: Bbox) -> int:
    return xxhash.xxh3_64_intdigest(f"{slide},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
