from PIL import Image
import math


# def f_to_u8(image: np.ndarray) -> np.ndarray:
#     return np.around(image).astype(np.uint8)


# def u8_to_f(image: np.ndarray) -> np.ndarray:
#     return image.astype(np.float32)


# def rgb2gray(image: np.ndarray) -> np.ndarray:
#     return f_to_u8(np.mean(image, axis=(2)))


# def minmax_norm(
#     image: np.ndarray,
#     min_low_bound: int = 0,
#     min_high_bound: int = 255,
#     max_low_bound: int = 0,
#     max_high_bound: int = 255,
# ) -> np.ndarray:
#     # clamp min, max value
#     min_val = max(min(np.min(image), min_high_bound), min_low_bound)
#     max_val = max(min(np.max(image), max_high_bound), max_low_bound)
#     image = image.clip(min_val, max_val)
#     image -= min_val
#     scale = 255 / (max_val - min_val)
#     image = f_to_u8(u8_to_f(image) * scale)
#     return image

#
# def rgb2gray(image: cv2.UMat) -> cv2.UMat:
#     r, g, b = cv2.split(cv2.multiply(image, 1 / 3))  # type: ignore
#     return cv2.add(cv2.add(r, g), b)
#
#
#
# def clip_scale_median(image: np.ndarray, info: dict) -> np.ndarray:
#     '''suppress backgroud noise and hue by clip and scale image color'''
#     median = np.array([info['r_median'], info['g_median'],
#                       info['b_median']], dtype=np.float32)
#     image = image.clip(None, median)
#     image = image * (255 / median)
#
#     return image


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
