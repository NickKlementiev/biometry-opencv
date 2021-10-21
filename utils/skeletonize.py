import numpy as np
import cv2 as cv
from utils.crossing_number import calculate_minutiaes
from skimage.morphology import skeletonize as skelt
from skimage.morphology import thin


def skeletonize(img_input):
    img = np.zeros_like(img_input)
    img[img_input == 0] = 1.0
    output = np.zeros_like(img_input)

    skeleton = skelt(img)

    output[skeleton] = 255
    cv.bitwise_not(output, output)
    
    return output

def thinning_morph(img, kernel):
    thinning_img = np.zeros_like(img)
    img_copy = img.copy()

    while 1:
        erosion = cv.erode(img_copy, kernel, iterations=1)
        dilatation = cv.dilate(erosion, kernel, iterations=1)

        subs_img = np.subtract(img, dilatation)
        cv.bitwise_or(thinning_img, subs_img, thinning_img)
        img_copy = erosion.copy()

        done = (np.sum(img_copy) == 0)

        if done:
            break

    down = np.zeros_like(thinning_img)
    down[1: -1, :] = thinning_img[0: -2, ]
    down_mask = np.subtract(down, thinning_img)
    down_mask[0: -2, :] = down_mask[1: -1, ]
    cv.imshow('down', down_mask)

    left = np.zeros_like(thinning_img)
    left[:, 1: -1] = thinning_img[:, 0: -2]
    left_mask = np.subtract(left, thinning_img)
    left_mask[:, 0: -2] = left_mask[:, 1: -1]
    cv.imshow('left', left_mask)

    cv.bitwise_or(down_mask, down_mask, thinning_img)
    output = np.zeros_like(thinning_img)
    output[thinning_img < 250] = 255

    return output
