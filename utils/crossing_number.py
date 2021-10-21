import cv2 as cv
import numpy as np

def minutiae_at(pixels, i, j, kernel_size):
    if pixels[i][j] == 1:
        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1),
                    (0, 1), (1, 1), (1, 0),
                    (1, -1), (0, -1), (-1, -1)]
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                    (-1, 2), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0),
                    (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]

        values = [pixels[i + l][j + k] for k, l in cells]

        crossings = 0

        for k in range(0, len(values) - 1):
            crossings += abs(values[k] - values[k + 1])
        crossings //= 2

        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"

def calculate_minutiaes(img, kernel_size=3):
    binary_img = np.zeros_like(img)
    binary_img[img < 10] = 1.0
    binary_img = binary_img.astype(np.int8)

    (y, x) = img.shape
    result = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}  # ending -> blue ; bifurcation -> green (BGR)

    for i in range(1, x - kernel_size // 2):
        for j in range(1, y - kernel_size // 2):
            minutiae = minutiae_at(binary_img, j, i, kernel_size)
            if minutiae != "none":
                cv.circle(result, (i, j), radius=1, color=colors[minutiae], thickness=1)

    return result
