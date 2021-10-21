from glob import glob

import numpy as np
import cv2 as cv
import sys

from fingerprint import fingerprint


def match(fingerprint1, fingerprint2):
    input1 = cv.imread(fingerprint1, cv.IMREAD_GRAYSCALE)
    results1 = fingerprint(input1)

    input2 = cv.imread(fingerprint2, cv.IMREAD_GRAYSCALE)
    results2 = fingerprint(input2)

    img1 = results1[6]
    img2 = results2[6]

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) >= 75:
        return "true"

    return "false"


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print(match(str(sys.argv[1]), str(sys.argv[2])))
    else:
        print("Usage: fingerprint_match.py [fingerprint1] [fingerprint2]")
