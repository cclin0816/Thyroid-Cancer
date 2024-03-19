import cv2
import math
import numpy as np
from PIL import Image

def ptc_fit_ellipse(contour):
    if len(contour) < 5 or len(contour) > 67:
        return
    ellipse = cv2.fitEllipse(contour)
    (w, h) = ellipse[1]
    if math.isnan(w) or w <= 13 or w >= 60:
        return
    if h <= 23 or h >= 118:
        return
    return ellipse


def ptc_find_template(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        ellipse = ptc_fit_ellipse(contour)
        if not ellipse == None:
            result.append(ellipse)
    return result


def bfc_fit_ellipse(contour):
    if len(contour) < 5 or len(contour) > 75:
        return
    ellipse = cv2.fitEllipse(contour)
    (w, h) = ellipse[1]
    if math.isnan(w) or w <= 15 or w >= 40:
        return
    if h <= 15 or h >= 40:
        return
    return ellipse


def bfc_find_template(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        ellipse = bfc_fit_ellipse(contour)
        if not ellipse == None:
            result.append(ellipse)
    return result

def find_cell(image: Image.Image, tag: str):
    if tag == 'PTC':
        return ptc_find_template(image)
    elif tag == 'BFC':
        return bfc_find_template(image)
    raise ValueError("bad tag")