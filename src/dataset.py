import utils
import math
import pandas as pd
import numpy as np
import openslide
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm


def pts_to_bb(points: list[list[float]]) -> utils.Bbox:
    ''' return bounding box that contains all points (inclusive on border) '''

    x_min = math.floor(min(points, key=lambda x: x[0])[0])
    x_max = math.ceil(max(points, key=lambda x: x[0])[0])
    y_min = math.floor(min(points, key=lambda x: x[1])[1])
    y_max = math.ceil(max(points, key=lambda x: x[1])[1])

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def mass_target_info(slide: str, targets: list[dict]) -> list[list]:
    info_df = []
    expect_fields = ('labels', 'segments', 'tool')

    for target in targets:

        if len(target['labels']) != 1:
            utils.log(
                'WARN', f"'{slide}' target skipped: no or multiple labels: {target['labels']}")
            continue

        match target['labels'][0]:
            case 'MMH_cytology_admin:HurthleCell':
                tag = 'Hurthle'
            case 'MMH_cytology_admin:Histiocytes':
                tag = 'Histiocyte'
            case 'MMH_cytology_admin:PTC':
                tag = 'PTC'
            case 'MMH_cytology_admin:FCs':
                tag = 'BFC'
            case 'MMH_cytology_admin:Indeterminate':
                tag = 'Indeterminate'
            case _:
                utils.log(
                    'WARN', f"'{slide}' target skipped: unknown label: '{target['labels'][0]}'")
                continue

        segments = target['segments']
        match target['tool']:
            case 'closed-path':
                bbox = pts_to_bb(segments)
            case 'rectangle':
                bbox = pts_to_bb(segments)
            case None:
                utils.log(
                    'NOTICE', f"'{slide}' target: no tool specified")
                bbox = pts_to_bb(segments)
            case _:
                utils.log(
                    'WARN', f"'{slide}' target skipped: unknown tool: '{target['tool']}'")
                continue

        for field in target.keys():
            if field not in expect_fields:
                utils.log(
                    'NOTICE', f"'{slide}' target: unknown field: '{field}': '{target[field]}'")

        uid = utils.mass_hash(slide, bbox)
        info_df.append([uid, slide, tag,
                       bbox[0], bbox[1], bbox[2], bbox[3]])

    return info_df


def get_mass_info(labels: list[dict]) -> pd.DataFrame:

    info_df = []
    expect_fields = ('slide_name', 'slide_is_ready',
                     'sub_tags', 'description', 'note', 'targets', 'project')

    for label in tqdm(labels):
        slide = label['slide_name']

        if not label['slide_is_ready']:
            utils.log('WARN', f"'{slide}' skipped: not ready")
            continue

        if label['description'] != None and label['description'] != '':
            utils.log(
                'NOTICE', f"'{slide}': ignore description: '{label['description']}'")

        if label['note'] != None and label['note'] != '':
            utils.log(
                'NOTICE', f"'{slide}': ignore note: '{label['note']}'")

        tags = label['sub_tags']
        if len(tags) > 1 or (len(tags) == 1 and tags[0] != 'done'):
            utils.log(
                'NOTICE', f"'{slide}': ignore tags: '{tags}'")

        for field in label.keys():
            if field not in expect_fields:
                utils.log(
                    'NOTICE', f"'{slide}': unknown field: '{field}': '{label[field]}'")

        info_df.extend(mass_target_info(slide, label['targets']))

    info_df = pd.DataFrame(info_df, columns=[
        'uid',
        'slide',
        'tag',
        'x_min',
        'y_min',
        'width',
        'height'])
    info_df.set_index('uid', inplace=True)
    info_df.drop_duplicates(inplace=True)
    conflict = info_df[info_df.index.duplicated(keep=False)]
    if not conflict.empty:
        utils.log('ERR', f"conflicts in dataset:\n{conflict}")

    return info_df


def slide_info(path: Path) -> list | None:
    try:
        slide = openslide.open_slide(path)
    except Exception as e:
        utils.log('WARN', f"'{path.stem}' skipped: openslide failed: {e}")
        return None

    # level 4 (x16) will have good enough proximation
    try:
        image = slide.read_region(
            (0, 0), 4, slide.level_dimensions[4])  # type: ignore
    except Exception as e:
        utils.log('WARN', f"'{path.stem}' skipped: read_region failed: {e}")
        return None

    image = np.array(image.convert("RGB"))
    p10 = np.percentile(image, 10, axis=(0, 1), method='closest_observation')
    p0 = np.amin(image, axis=(0, 1))
    return [path.stem, *p10, *p0]


def get_slide_info(path_list: list[Path]) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = tqdm(executor.map(slide_info, path_list),
                      total=len(path_list))
        info_df = pd.DataFrame(
            filter(lambda x: x is not None, result),  # type: ignore
            columns=['name', 'r_p10', 'g_p10',
                     'b_p10', 'r_min', 'g_min', 'b_min']
        )
        info_df.set_index('name', inplace=True, verify_integrity=True)

        return info_df
