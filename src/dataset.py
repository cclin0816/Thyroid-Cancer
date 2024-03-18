import utils
import math
import pandas as pd
import numpy as np
import openslide
from pathlib import Path


def minmax_to_bb(x_min: int, x_max: int, y_min: int, y_max: int) -> utils.Bbox:
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def pts_to_bb(points: list[list[float]]) -> utils.Bbox:
    ''' return bounding box that contains all points (inclusive on border) '''

    x_min = math.floor(min(points, key=lambda x: x[0])[0])
    x_max = math.ceil(max(points, key=lambda x: x[0])[0])
    y_min = math.floor(min(points, key=lambda x: x[1])[1])
    y_max = math.ceil(max(points, key=lambda x: x[1])[1])

    return minmax_to_bb(x_min, x_max, y_min, y_max)


def mass_target_info(slide_name: str, targets: list[dict]) -> list[list]:
    info_df = []
    expect_fields = ('labels', 'segments', 'tool')

    for target in targets:

        if len(target['labels']) != 1:
            utils.log(
                'WARN', f"'{slide_name}' target skipped: no or multiple labels: {target['labels']}")
            continue

        mass_class = None
        match target['labels'][0]:
            case 'MMH_cytology_admin:HurthleCell':
                mass_class = 'Hurthle'
            case 'MMH_cytology_admin:Histiocytes':
                mass_class = 'Histiocyte'
            case 'MMH_cytology_admin:PTC':
                mass_class = 'PTC'
            case 'MMH_cytology_admin:FCs':
                mass_class = 'FC'
            case 'MMH_cytology_admin:Indeterminate':
                mass_class = 'Indeterminate'
            case _:
                utils.log(
                    'WARN', f"'{slide_name}' target skipped: unknown label: '{target['labels'][0]}'")
                continue

        bbox = None
        segments = target['segments']
        match target['tool']:
            case 'closed-path':
                bbox = pts_to_bb(segments)
            case 'rectangle':
                bbox = pts_to_bb(segments)
            case None:
                utils.log(
                    'NOTICE', f"'{slide_name}' target: no tool specified")
                bbox = pts_to_bb(segments)
            case _:
                utils.log(
                    'WARN', f"'{slide_name}' target skipped: unknown tool: '{target['tool']}'")
                continue

        for field in target.keys():
            if field not in expect_fields:
                utils.log(
                    'NOTICE', f"'{slide_name}' target: unknown field: '{field}': '{target[field]}'")

        uid = hash((slide_name, bbox))
        info_df.append([uid, slide_name, mass_class,
                       bbox[0], bbox[1], bbox[2], bbox[3]])

    return info_df


def get_mass_info(labels: list[dict]) -> pd.DataFrame:

    info_df = []
    expect_fields = ('slide_name', 'slide_is_ready',
                     'sub_tags', 'description', 'note', 'targets', 'project')

    for label in labels:
        slide_name = label['slide_name']

        if not label['slide_is_ready']:
            utils.log('WARN', f"'{slide_name}' skipped: not ready")
            continue

        if label['description'] != None and label['description'] != '':
            utils.log(
                'NOTICE', f"'{slide_name}': ignore description: '{label['description']}'")

        if label['note'] != None and label['note'] != '':
            utils.log(
                'NOTICE', f"'{slide_name}': ignore note: '{label['note']}'")

        tags = label['sub_tags']
        if len(tags) > 1 or (len(tags) == 1 and tags[0] != 'done'):
            utils.log(
                'NOTICE', f"'{slide_name}': ignore tags: '{tags}'")

        for field in label.keys():
            if field not in expect_fields:
                utils.log(
                    'NOTICE', f"'{slide_name}': unknown field: '{field}': '{label[field]}'")

        info_df.extend(mass_target_info(slide_name, label['targets']))

        utils.log('INFO', f"'{slide_name}' finished")

    info_df = pd.DataFrame(info_df, columns=[
        'uid',
        'slide',
        'class',
        'x_min',
        'y_min',
        'width',
        'height'])
    info_df.set_index('uid', inplace=True)
    info_df.drop_duplicates(inplace=True)
    conflict = info_df[info_df.index.duplicated(keep=False)]
    if not conflict.empty:
        utils.log('WARN', f"conflicts in dataset:\n{conflict}")

    return info_df


def get_slide_info(path_list: list[Path]) -> pd.DataFrame:
    info_df = []

    for path in path_list:
        slide = None
        try:
            slide = openslide.open_slide(path)
        except Exception as e:
            utils.log(
                'WARN', f"'{path.stem}' skipped: openslide failed: {e}")
            continue

        info = []
        # name
        info.append(path.stem)

        # level 4 (x16) will have good enough proximation
        img = slide.read_region(
            (0, 0), 4, slide.level_dimensions[4])  # type: ignore
        img = np.array(img)

        # rgb
        for idx in range(3):
            channel = img[..., idx]
            # channel min
            info.append(np.amin(channel))
            # channel median
            info.append(np.median(channel).astype(np.uint8))

        info_df.append(info)

        utils.log('INFO', f"'{path.stem}' finished")

    info_df = pd.DataFrame(info_df, columns=[
        'name',
        'r_min',
        'r_median',
        'g_min',
        'g_median',
        'b_min',
        'b_median',
    ])
    info_df.set_index('name', inplace=True, verify_integrity=True)

    return info_df
