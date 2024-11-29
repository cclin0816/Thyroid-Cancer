import math
import cv2
import numpy
from PIL import Image


def resize_keep_ratio(image: Image.Image, width: int, height: int) -> Image.Image:
    # image.thumbnail() fails when image is smaller than width, height
    ratio = image.width / image.height
    if ratio > 1.0:
        w = width
        h = round(w / ratio)
    else:
        h = height
        w = round(h * ratio)

    return image.resize((w, h), Image.Resampling.LANCZOS)


def grid_composite(
    images: list[Image.Image], row_len: int = 16, width: int = 100, height: int = 100
) -> Image.Image:
    col_len = math.ceil(len(images) / row_len)
    grid = Image.new("RGB", (width * row_len, height * col_len))

    for idx, image in enumerate(images):
        img = resize_keep_ratio(image, width, height)
        x = (idx % row_len) * width + (width - img.width) // 2
        y = (idx // row_len) * height + (height - img.height) // 2
        grid.paste(img, (x, y))

    return grid


def crop_rotated_rectangle(
    image: numpy.ndarray,
    center: tuple[float, float],
    size: tuple[int, int],
    angle: float,
) -> numpy.ndarray:
    # crop small square first to speed up rotate
    (w, h) = size
    cs = math.ceil(math.sqrt((w / 2) ** 2 + (h / 2) ** 2) * 2)
    cc = (cs - 1) / 2
    cs = (cs, cs)
    cc = (cc, cc)

    crop = cv2.getRectSubPix(image, cs, center)
    rot_mat = cv2.getRotationMatrix2D(cc, angle, 1)
    rotate = cv2.warpAffine(crop, rot_mat, cs)
    return cv2.getRectSubPix(rotate, size, cc)


# def histogram(image: np.ndarray) -> None:
#     '''plot color intensity excluding 255'''
#     plt.figure()
#     colors = ('r', 'g', 'b')
#     for idx, color in enumerate(colors):
#         hist = cv2.calcHist([image], [idx], None,
#                             [256], [0, 256])[..., 0]
#         hist[255] = hist[254]
#         for idx in range(1, 254):
#             if hist[idx] == 0:
#                 hist[idx] = (hist[idx - 1] + hist[idx + 1]) / 2
#         plt.plot(hist, color=color)
#         plt.xlim([0, 256])
#     plt.show()
# def histogram_gray(image: np.ndarray) -> None:
#     '''plot color intensity excluding 255'''
#     plt.figure()
#     hist = cv2.calcHist([image], [0], None,
#                         [256], [0, 256])[..., 0]
#     plt.plot(hist)
#     plt.xlim([0, 256])
#     plt.show()
