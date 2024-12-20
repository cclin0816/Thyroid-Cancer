import cv2
import math
import numpy as np

import utils


ones3x3 = np.ones((3, 3), dtype=np.uint8)


def fit_ellipse(contour):
    # not enough point to fit or to way too brittle
    length = len(contour)
    if length < 5 or length > 120:
        return

    elps = cv2.fitEllipse(contour)
    (cx, cy) = elps[0]
    (w, h) = elps[1]
    # bad fit
    if any(map(math.isnan, (cx, cy, w, h))):
        return
    # check if size of ellipse is valid
    if not (15 < w < 60 and 20 < h < 120 and h / w < 2.5):
        return

    return elps


def find_cell(
    slide: str, bbox: utils.Bbox, reader: utils.SlideReader, info: dict
) -> list[tuple]:
    image = reader.read_bbox(slide, bbox)
    image = np.array(image.convert("RGB"))

    # suppress backgroud
    p10 = np.array([info["r_p10"], info["g_p10"], info["b_p10"]])
    p0 = np.array([info["r_min"], info["g_min"], info["b_min"]])
    scale = 255 / (p10 - p0).astype(np.float32)
    image = (np.clip(image, None, p10) - p0).astype(np.float32)
    image = image * scale

    # generate mask, opencv adaptive threshold but for float32
    gray = np.mean(image, axis=2)
    blur = cv2.GaussianBlur(gray, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)
    thres = cv2.GaussianBlur(gray, (21, 21), 0, borderType=cv2.BORDER_REPLICATE) - 1
    mask = cv2.compare(blur, thres, cv2.CMP_LE)
    # cleanup mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ones3x3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ones3x3)

    contours, hierarchies = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return []

    result = []
    for contour, hierarchy in zip(contours, hierarchies[0]):
        # outer region only, see cv2.RETR_CCOMP
        if hierarchy[3] != -1:
            continue
        elps = fit_ellipse(contour)
        if elps is None:
            continue

        center, size, angle = elps
        score = cell_score(image, elps)
        result.append((*center, *size, angle, score))

    return result


def cell_score(image, elps):
    ((cx, cy), (w, h), angle) = elps
    w = math.ceil(w)
    h = math.ceil(h)
    # out crop a little
    cw = w + 6
    ch = h + 6

    # check cell on edge
    (ih, iw, _) = np.shape(image)
    r = ch / 2
    if cx - r < 0 or cx + r > iw - 1 or cy - r < 0 or cy + r > ih - 1:
        return 0.0

    crop = utils.crop_rotated_rectangle(image, (cx, cy), (cw, ch), angle)

    cc = ((cw - 1) / 2, (ch - 1) / 2)
    cell_mask = cv2.ellipse(
        np.zeros((ch, cw), dtype=np.uint8),
        box=(cc, (w, h), 0),
        color=255,  # type: ignore
        thickness=-1,
    )
    rim_mask = cv2.ellipse(
        np.zeros((ch, cw), dtype=np.uint8),
        box=(cc, (cw, ch), 0),
        color=255,  # type: ignore
        thickness=-1,
    )
    rim_mask ^= cell_mask
    cell_mask = cell_mask == 0
    rim_mask = rim_mask == 0

    # adjust contrast
    cell = np.copy(crop)
    cell[np.repeat(cell_mask[..., np.newaxis], 3, 2)] = np.nan
    rim = np.copy(crop)
    rim[np.repeat(rim_mask[..., np.newaxis], 3, 2)] = np.nan

    clip_min = np.nanpercentile(cell, 30, axis=(0, 1))
    clip_max = np.nanpercentile(rim, 70, axis=(0, 1))
    # bad contrast
    if np.any(clip_min >= clip_max):
        return 0.0

    crop = np.clip(crop, clip_min, clip_max) - clip_min
    crop = crop * 255 / (clip_max - clip_min)
    crop = np.mean(crop, axis=2)

    # score by magic method that LGTM
    cell = np.copy(crop)
    cell[cell_mask] = np.nan
    rim = np.copy(crop)
    rim[rim_mask] = np.nan

    cell_thres = np.nanpercentile(cell, 90)
    rim_thres = np.nanpercentile(rim, 30)
    score = min(max((rim_thres - cell_thres) * 2, 0), 40) * 1.5  # type: ignore
    score += min(max(80 - np.nanstd(cell), 0), 40)  # type: ignore
    score += min(max(80 - np.nanmean(cell), 0), 40)  # type: ignore

    return score / 140
