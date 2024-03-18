from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def scale_color(img: np.ndarray, ratio: float) -> np.ndarray:
    return np.around(img.astype(np.float32) * ratio).astype(np.uint8)


def suppress_bg(img: Image.Image, info: dict) -> Image.Image:
    '''suppress backgroud noise and hue by clip and scale image color'''
    colors = ('r', 'g', 'b')
    img_data = np.array(img)

    for idx, color in enumerate(colors):
        channel = img_data[..., idx]
        # 6 and 24 are chosen by lgtm on slide histograms
        # so the color intensity distribution would be more even
        clip_low = info[color + '_min'] + 6
        clip_high = info[color + '_median'] - 24
        channel = channel.clip(clip_low, clip_high)
        channel -= clip_low
        channel = scale_color(channel, 255 / (clip_high - clip_low))
        img_data[..., idx] = channel

    return Image.fromarray(img_data)


def histogram(img: Image.Image) -> None:
    '''plot color intensity excluding 255'''
    colors = ('r', 'g', 'b')
    img_data = np.array(img)
    plt.figure()

    for idx, color in enumerate(colors):
        hist = cv2.calcHist([img_data], [idx], None,
                            [256], [0, 256])[..., 0]

        hist[255] = hist[254]
        for idx in range(1, 254):
            if hist[idx] == 0:
                hist[idx] = (hist[idx - 1] + hist[idx + 1]) / 2

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()


def resize_keep_ratio(image: Image.Image, width: int, height: int) -> Image.Image:
    # image.thumbnail() fails when image is smaller than width, height
    ratio = image.width / image.height
    w = None
    h = None
    if ratio > 1.0:
        w = width
        h = round(w / ratio)
    else:
        h = height
        w = round(h * ratio)

    return image.resize((w, h), Image.LANCZOS)


def grid_composite(images: list[Image.Image], row_len: int = 16, width: int = 100, height: int = 100) -> Image.Image:
    col_len = math.ceil(len(images) / row_len)
    grid = Image.new(
        "RGB", (width * row_len, height * col_len))

    for idx, image in enumerate(images):
        img = resize_keep_ratio(image, width, height)
        x = (idx % row_len) * width + (width - img.width) // 2
        y = (idx // row_len) * height + (height - img.height) // 2
        grid.paste(img, (x, y))

    return grid
