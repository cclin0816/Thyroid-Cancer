import cv2
import math
import numpy as np
from pandas import Series

import utils


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


# def crop_rotate(image, center, size, angle):
#     # crop small square first to speed up rotate
#     (w, h) = size
#     crop_size = math.ceil(math.sqrt((w / 2) ** 2 + (h / 2) ** 2) * 2)
#     crop = cv2.getRectSubPix(image, (crop_size, crop_size), center)
#     crop_center = crop_size / 2

#     rot_mat = cv2.getRotationMatrix2D((crop_center, crop_center), angle, 1)
#     rotate = cv2.warpAffine(crop, rot_mat, (crop_size, crop_size))

#     crop = cv2.getRectSubPix(rotate, (w, h), (crop_center, crop_center))

#     return crop


# def bg_mean(image):
#     (h, w) = np.shape(image)
#     center_mask = np.zeros((h, w), dtype=bool)
#     center_mask[9:(h - 9), 9:(w - 9)
#                 ] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w - 18, h - 18))
#     bg = np.ma.array(image, mask=center_mask)
#     return np.ma.mean(bg)


# def refine_cell(image, elps):
#     ((cx, cy), (w, h), angle) = elps
#     w = math.ceil(w)
#     h = math.ceil(h)
#
#     # eliminate partial cell (ellipse on edges)
#     (img_h, img_w, _) = np.shape(image)
#     r = h / 2
#     if cx - r < 0 or cx + r + 1 > img_w or cy - r < 0 or cy + r + 1 > img_h:
#         return []
#
#     crop = crop_rotate(image, (cx, cy), (w + 15, h + 15), angle)
#     for idx in range(3):
#         crop[..., idx] = imt.minmax_norm(crop[..., idx])
#     gray = imt.rgb2gray(crop)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     thres = bg_mean(blur)
#     edge = cv2.Canny(blur, thres * 0.66, thres * 1.33)
#     darker = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)
#     darker = cv2.morphologyEx(darker, cv2.MORPH_OPEN, np.array(
#         [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
#
#     bd.display(Image.fromarray(blur))
#     bd.display(Image.fromarray(edge))
#     bd.display(Image.fromarray(darker))
#
#     contours, hierarchies = cv2.findContours(
#         darker, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     # no contour found
#     if hierarchies is None:
#         return []
#
#     for contour, hierarchy in zip(contours, hierarchies[0]):
#         # outer region only, see cv2.RETR_CCOMP
#         if hierarchy[3] != -1:
#             continue
#         segment = cv2.fillPoly(
#             # type: ignore
#             np.zeros((h + 15, w + 15), dtype=np.uint8), [contour], 255)
#         segment = cv2.dilate(segment, np.ones(
#             (3, 3), dtype=np.uint8)) & edge  # type: ignore
#
#         bd.display(Image.fromarray(segment))
#
#     bd.flush()
#
#     return [elps]


ones3x3 = np.ones((3, 3), dtype=np.uint8)


def find_cell(index, value: Series, reader: utils.SlideReader, info: dict) -> list:
    ori_image = reader.read_bbox(value["slide"], utils.get_bbox(value))  # type: ignore

    p10 = np.array([info["r_p10"], info["g_p10"], info["b_p10"]])
    p0 = np.array([info["r_min"], info["g_min"], info["b_min"]])
    scale = 255 / (p10 - p0).astype(np.float32)

    image = ori_image.convert("RGB")
    image = np.array(image)
    image = (np.clip(image, None, p10) - p0).astype(np.float32)
    image = image * scale

    gray = np.mean(image, axis=2)
    blur = cv2.GaussianBlur(gray, (11, 11), 0, borderType=cv2.BORDER_REPLICATE)
    thres = cv2.GaussianBlur(gray, (21, 21), 0, borderType=cv2.BORDER_REPLICATE) - 1

    mask = cv2.compare(blur, thres, cv2.CMP_LE)
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

        # result.extend(refine_cell(image, elps))

        center, size, angle = elps
        result.append((index, *center, *size, angle))

    return result
