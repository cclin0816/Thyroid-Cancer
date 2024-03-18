from pathlib import Path
import json
from typing import Any
import sys


def load_json(path: Path) -> Any:
    with path.open('r') as fp:
        return json.load(fp)


def dump_json(path: Path, obj: Any):
    with path.open('w') as fp:
        json.dump(obj, fp)


cur_log_lvl = 3
log_lvl_lut = {'DEBUG': 5, 'INFO': 4,
               'NOTICE': 3, 'WARN': 2, 'ERR': 1, 'CRIT': 0}


def log(lvl: str, msg: str):
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl == None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    if log_lvl <= cur_log_lvl:
        print(f"[{lvl}] {msg}", file=sys.stderr)


def set_log_lvl(lvl: str):
    log_lvl = log_lvl_lut.get(lvl)
    if log_lvl == None:
        raise ValueError(f"bad lvl: '{lvl}'")

    global cur_log_lvl
    cur_log_lvl = log_lvl


Bbox = tuple[int, int, int, int]
''' type alias of (x_min, y_min, width, height) '''
