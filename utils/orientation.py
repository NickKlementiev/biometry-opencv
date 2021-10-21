import math
import numpy as np
import cv2 as cv

def calculate_angles(img, W, smooth=False):
    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2

    (y, x) = img.shape

    sobel_operator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    y_sobel = np.array(sobel_operator).astype(np.int)
    x_sobel = np.transpose(y_sobel).astype(np.int)

    result = [[] for i in range(1, y, W)]

    Gx_ = cv.filter2D(img / 125, -1, y_sobel) * 125
    Gy_ = cv.filter2D(img / 125, -1, x_sobel) * 125
            
    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0
            denominator = 0
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W, x - 1)):
                    Gx = round(Gx_[l, k])
                    Gy = round(Gy_[l, k])
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)
            
            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                orientation = np.pi / 2 + math.atan2(nominator, denominator) / 2
                result[int((j - 1) // W)].append(angle)
            else:
                result[int((j - 1) // W)].append(0)

    result = np.array(result)

    if smooth:
        result = smooth_angles(result)

    return result

def gauss(x, y):
    sigma = 1.0
    return (1 / (2 * math.pi * sigma)) * math.exp(-(x * x + y * y)) / (2 * sigma)

def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))

    return kernel

def smooth_angles(angles):
    angles = np.array(angles)
    cos_angles = np.cos(angles.copy() * 2)
    sin_angles = np.sin(angles.copy()* 2)

    kernel = np.array(kernel_from_function(5, gauss))

    cos_angles = cv.filter2D(cos_angles / 125, -1, kernel) * 125
    sin_angles = cv.filter2D(sin_angles / 125, -1, kernel) * 125
    smooth_angles = np.arctan2(sin_angles, cos_angles) / 2

    return smooth_angles

def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W / 2) * tang + j + W / 2))
        end = (i + W, int((W / 2) * tang + j + W / 2))
    else:
        begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
        end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)

    return (begin, end)

def visualize_angles(img, mask, angles, W):
    (y, x) = img.shape
    result = cv.cvtColor(np.zeros(img.shape, np.uint8), cv.COLOR_GRAY2RGB)
    mask_threshold = (W - 1) ** 2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1: j + W, i - 1: i + W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = get_line_ends(i, j, W, tang)
                cv.line(result, begin, end, color=150)

    cv.resize(result, img.shape, result)
    return result
                
            