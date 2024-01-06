from pathlib import Path
import utils
import math


def minmax_to_bbox(x_min: int, x_max: int, y_min: int, y_max: int) -> (int, int, int, int):
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def rectangle(segments: list[list[float]]) -> (int, int, int, int):
    if len(segments) != 4:
        return None
    for seg in segments:
        if len(seg) != 2:
            return None

    segments.sort(key=lambda x: x[1])
    segments.sort(key=lambda x: x[0])

    if segments[0][0] != segments[1][0]:
        return None
    if segments[2][0] != segments[3][0]:
        return None
    if segments[0][1] != segments[2][1]:
        return None
    if segments[1][1] != segments[3][1]:
        return None

    x_min = math.floor(segments[0][0])
    x_max = math.ceil(segments[2][0])
    y_min = math.floor(segments[0][1])
    y_max = math.ceil(segments[1][1])

    return minmax_to_bbox(x_min, x_max, y_min, y_max)


def closed_path(segments: list[list[float]]) -> (int, int, int, int):
    x_min = math.floor(min(segments, key=lambda x: x[0])[0])
    x_max = math.ceil(max(segments, key=lambda x: x[0])[0])
    y_min = math.floor(min(segments, key=lambda x: x[1])[1])
    y_max = math.ceil(max(segments, key=lambda x: x[1])[1])

    return minmax_to_bbox(x_min, x_max, y_min, y_max)


def fix_targets(sname: str, targets: list[dict]) -> list[dict]:
    new_targets = []

    for target in targets:
        cluster_type = None
        if len(target['labels']) != 1:
            utils.log(
                'CRIT', f"'{sname}' target skipped: unknown labels '{target['labels']}'")
            continue

        match target['labels'][0]:
            case 'MMH_cytology_admin:HurthleCell':
                cluster_type = 'Hurthle'
            case 'MMH_cytology_admin:Histiocytes':
                cluster_type = 'Histiocyte'
            case 'MMH_cytology_admin:PTC':
                cluster_type = 'PTC'
            case 'MMH_cytology_admin:FCs':
                cluster_type = 'FC'
            case 'MMH_cytology_admin:Indeterminate':
                cluster_type = 'Indeterminate'
            case _:
                utils.log(
                    'CRIT', f"'{sname}' target skipped: unknown label '{target['labels'][0]}'")
                continue

        bbox = None
        segments = target['segments']
        match target['tool']:
            case 'closed-path':
                bbox = closed_path(segments)
            case 'rectangle':
                bbox = rectangle(segments)
            case None:
                utils.log('INFO', 'no tool: auto detect')
                if len(segments) == 4:
                    bbox = rectangle(segments)
                else:
                    bbox = closed_path(segments)
            case _:
                utils.log(
                    'CRIT', f"'{sname}' target skipped: unknown tool '{target['tool']}'")
                continue

        if bbox == None:
            utils.log(
                'CRIT', f"'{sname}' target skipped: bad bbox: {segments}")
            continue

        expect_fields = ['labels', 'segments', 'tool']
        for field in target.keys():
            if field not in expect_fields:
                utils.log(
                    'WARN', f"'{sname}' unknown field: '{field}': '{target[field]}'")

        new_targets.append({'type': cluster_type, 'bbox_top_left_w_h': bbox})

    return new_targets


def fix_labels(labels: list[dict]) -> list[dict]:
    new_labels = []

    for label in labels:
        sname = label['slide_name']

        if not label['slide_is_ready']:
            utils.log('CRIT', f"'{sname}' skipped: not ready")
            continue

        if label['description'] != None and label['description'] != '':
            utils.log(
                'WARN', f"'{sname}' contains description: '{label['description']}'")

        if label['note'] != None and label['note'] != '':
            utils.log(
                'WARN', f"'{sname}' contains note: '{label['note']}'")

        tags = label['sub_tags']
        if len(tags) > 1 or (len(tags) == 1 and tags[0] != 'done'):
            utils.log(
                'WARN', f"'{sname}' contains tags: '{tags}'")

        expect_fields = ['slide_name', 'slide_is_ready',
                         'sub_tags', 'description', 'note', 'targets', 'project']
        for field in label.keys():
            if field not in expect_fields:
                utils.log(
                    'WARN', f"'{sname}' contains field: '{field}': '{label[field]}'")

        new_labels.append({
            'slide_name': sname,
            'clusters': fix_targets(sname, label['targets']),
        })

    return new_labels


if __name__ == '__main__':
    utils.log_lvl = 2
    cwd = Path('./')
    old_labels = utils.load_json(cwd / 'dataset' / 'labels.json')
    new_labels = fix_labels(old_labels)
    utils.dump_json(cwd / 'output' / 'labels_fixed.json', new_labels)
