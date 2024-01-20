import utils
import math
import pandas as pd


def minmax_to_bb(x_min: int, x_max: int, y_min: int, y_max: int) -> utils.Bbox:
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def pts_to_bb(points: list[list[float]]) -> utils.Bbox:
    ''' return bounding box that contains all points (inclusive on border) '''

    x_min = math.floor(min(points, key=lambda x: x[0])[0])
    x_max = math.ceil(max(points, key=lambda x: x[0])[0])
    y_min = math.floor(min(points, key=lambda x: x[1])[1])
    y_max = math.ceil(max(points, key=lambda x: x[1])[1])

    return minmax_to_bb(x_min, x_max, y_min, y_max)


def fix_mass_label(slide_name: str, targets: list[dict], labels: list[list]) -> None:
    expect_fields = ('labels', 'segments', 'tool')

    for target in targets:

        if len(target['labels']) != 1:
            utils.log(
                'CRIT', f"'{slide_name}' target skipped: unknown labels: '{target['labels']}'")
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
                    'CRIT', f"'{slide_name}' target skipped: unknown label: '{target['labels'][0]}'")
                continue

        bbox = None
        segments = target['segments']
        match target['tool']:
            case 'closed-path':
                bbox = pts_to_bb(segments)
            case 'rectangle':
                bbox = pts_to_bb(segments)
            case None:
                utils.log('INFO', f"'{slide_name}' target: no tool specified")
                bbox = pts_to_bb(segments)
            case _:
                utils.log(
                    'CRIT', f"'{slide_name}' target skipped: unknown tool: '{target['tool']}'")
                continue

        for field in target.keys():
            if field not in expect_fields:
                utils.log(
                    'WARN', f"'{slide_name}' target: unknown field: '{field}': '{target[field]}'")

        uid = hash((slide_name, bbox))
        labels.append([uid, slide_name, mass_class,
                       bbox[0], bbox[1], bbox[2], bbox[3]])


def fix_labels(labels: list[dict], allowlist: set[str]) -> pd.DataFrame:
    label_names = set()
    new_labels = []
    expect_fields = ('slide_name', 'slide_is_ready',
                     'sub_tags', 'description', 'note', 'targets', 'project')

    for label in labels:
        slide_name = label['slide_name']

        if slide_name not in allowlist:
            utils.log('WARN', f"'{slide_name}' skipped: not in allowlist")
            continue

        if slide_name in label_names:
            utils.log('CRIT', f"'{slide_name}' skipped: duplicate label")
            continue
        label_names.add(slide_name)

        if not label['slide_is_ready']:
            utils.log('CRIT', f"'{slide_name}' skipped: not ready")
            continue

        if label['description'] != None and label['description'] != '':
            utils.log(
                'WARN', f"'{slide_name}': ignore description: '{label['description']}'")

        if label['note'] != None and label['note'] != '':
            utils.log(
                'WARN', f"'{slide_name}': ignore note: '{label['note']}'")

        tags = label['sub_tags']
        if len(tags) > 1 or (len(tags) == 1 and tags[0] != 'done'):
            utils.log(
                'WARN', f"'{slide_name}': ignore tags: '{tags}'")

        for field in label.keys():
            if field not in expect_fields:
                utils.log(
                    'WARN', f"'{slide_name}': unknown field: '{field}': '{label[field]}'")

        fix_mass_label(slide_name, label['targets'], new_labels)

    df = pd.DataFrame(new_labels, columns=[
        'uid',
        'slide',
        'class',
        'x_min',
        'y_min',
        'width',
        'height'])
    df.set_index('uid', inplace=True)
    return df
