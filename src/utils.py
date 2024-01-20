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


log_lvl = 4


def log(lvl: str, msg: str):
    global log_lvl

    if lvl == 'INFO' and log_lvl >= 4:
        print('[INFO] ' + msg, file=sys.stderr)
    elif lvl == 'WARN' and log_lvl >= 3:
        print('[WARN] ' + msg, file=sys.stderr)
    elif lvl == 'CRIT' and log_lvl >= 2:
        print('[CRIT] ' + msg, file=sys.stderr)
    elif lvl == 'ERRO' and log_lvl >= 1:
        print('[ERRO] ' + msg, file=sys.stderr)


Bbox = tuple[int, int, int, int]
''' type alias of (x_min, y_min, width, height) '''
