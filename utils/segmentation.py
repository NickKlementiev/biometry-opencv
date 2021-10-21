import numpy as np
import cv2 as cv


def normalise(img):
    return (img - np.mean(img))/(np.std(img))


def create_segmented_and_variance_images(img, w, threshold=.2):
    (y, x) = img.shape
    threshold = np.std(img)*threshold

    image_variance = np.zeros(img.shape)
    segmented_image = img.copy()
    mask = np.ones_like(img)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(img[box[1]: box[3], box[0]: box[2]])
            image_variance[box[1]: box[3], box[0]: box[2]] = block_stddev

    mask[image_variance < threshold] = 0

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w * 2, w * 2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    segmented_image *= mask
    img = normalise(img)
    mean_val = np.mean(img[mask == 0])
    std_val = np.std(img[mask == 0])
    norm_img = (img - mean_val) / std_val

    return segmented_image, norm_img, mask
