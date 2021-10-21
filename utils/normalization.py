from math import sqrt
import numpy as np


def normalize_pixel(x, v0, v, m, m0):
    dev_coeff = sqrt((v0 * ((x - m) ** 2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def normalize(img, m0, v0):
    m = np.mean(img)
    v = np.std(img) ** 2
    (y, x) = img.shape
    normalized_image = img.copy()
    
    for i in range(x):
        for j in range(y):
            normalized_image[j, i] = normalize_pixel(img[j, i], v0, v, m, m0)

    return normalized_image
