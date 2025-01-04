import cv2
import math
import numpy as np
import colorsys

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


def rescale(image, clip_min, clip_max):
    image = np.clip(image, clip_min, clip_max) - clip_min
    return image / (clip_max - clip_min)


def find_cell(
    slide: str, bbox: utils.Bbox, reader: utils.SlideReader, info: dict
) -> list[tuple]:
    image = reader.read_bbox(slide, bbox)
    image = np.array(image.convert("RGB"), dtype=np.float32)

    # suppress backgroud
    clip_max = np.array([info["r_p10"], info["g_p10"], info["b_p10"]], dtype=np.float32)
    clip_min = np.array([info["r_min"], info["g_min"], info["b_min"]], dtype=np.float32)
    image = rescale(image, clip_min, clip_max)

    # generate mask, adaptive threshold but for float32
    gray = np.mean(image, axis=2)
    blur = cv2.GaussianBlur(gray, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)
    thres = cv2.GaussianBlur(gray, (25, 25), 0, borderType=cv2.BORDER_REPLICATE) - 0.01
    mask = cv2.compare(blur, thres, cv2.CMP_LT)
    # cleanup mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ones3x3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ones3x3)

    # fit contour with ellipse
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
        if score == 0.0:
            continue
        result.append((*center, *size, angle, score))

    return result


def ellipse_mask(mw, mh, ew, eh):
    return (
        cv2.ellipse(
            np.zeros((mh, mw), dtype=np.uint8),
            box=(((mw - 1) / 2, (mh - 1) / 2), (ew, eh), 0),
            color=255,  # type: ignore
            thickness=-1,
        )
        == 0
    )


def mask_nan(image, mask):
    if image.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[..., np.newaxis], np.shape(image)[2], 2)
    image = np.copy(image)
    image[mask] = np.nan
    return image


def on_border(shape, x, y, r):
    (h, w, _) = shape
    return x - r < 0 or x + r > w - 1 or y - r < 0 or y + r > h - 1


def cell_score(image, elps):
    ((cx, cy), (w, h), angle) = elps
    w = math.ceil(w)
    h = math.ceil(h)
    # out crop a little
    cw = w + 6
    ch = h + 6

    # partial cell filter
    if on_border(np.shape(image), cx, cy, ch / 2):
        return 0.0

    crop = utils.crop_rotated_rectangle(image, (cx, cy), (cw, ch), angle)
    crop = cv2.GaussianBlur(crop, (7, 7), 0, borderType=cv2.BORDER_REPLICATE)

    cell_mask = ellipse_mask(cw, ch, w, h)
    cell = mask_nan(crop, cell_mask)

    # color filter
    (hue, light, saturate) = colorsys.rgb_to_hls(*np.nanmean(cell, (0, 1)))
    if hue < 0.55 or hue > 0.7 or light > 0.7 or saturate < 0.2:
        return 0.0

    rim_mask = ellipse_mask(cw, ch, cw, ch) | ~cell_mask
    rim = mask_nan(crop, rim_mask)

    # adjust contrast
    clip_min = np.nanpercentile(cell, 10, axis=(0, 1))
    clip_max = np.nanpercentile(rim, 30, axis=(0, 1))
    if np.any(clip_min >= (clip_max - 0.02)):
        return 0.0
    crop = rescale(crop, clip_min, clip_max)
    crop = np.mean(crop, axis=2)
    cell = mask_nan(crop, cell_mask)
    rim = mask_nan(crop, rim_mask)

    # separation score
    cell_thres = np.nanpercentile(cell, 95)
    rim_thres = np.nanpercentile(rim, 20)
    sep_score = utils.clamp(rim_thres - cell_thres, 0, 0.1) * 10
    if sep_score < 0.4:
        return 0.0

    stddev_1 = utils.clamp(0.35 - np.nanstd(cell), 0, 0.15) / 0.15
    if stddev_1 < 0.2:
        return 0.0

    crop = rescale(crop, cell_thres, rim_thres)
    cell = mask_nan(crop, cell_mask)
    stddev_2 = utils.clamp(0.3 - np.nanstd(cell), 0, 0.15) / 0.15
    if stddev_2 < 0.2:
        return 0.0

    return sep_score * stddev_1 * stddev_2
