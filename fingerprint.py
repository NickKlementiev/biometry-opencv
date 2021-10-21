import cv2 as cv
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize


def fingerprint(input_img):
    block_size = 16

    normalized_img = normalize(input_img.copy(), float(100), float(100))

    (segmented_img, norm_img, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    angles = orientation.calculate_angles(normalized_img, W=block_size, smooth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    freq = ridge_freq(norm_img, mask, angles, block_size, kernel_size=5, min_wave=5, max_wave=15)

    gabor_img = gabor_filter(norm_img, angles, freq)

    thin_img = skeletonize(gabor_img)

    minutiaes = calculate_minutiaes(thin_img)

    singularities_img = calculate_singularities(thin_img, angles, 1, block_size, mask)

    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_img, minutiaes, singularities_img]
    
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)

    results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

    return output_imgs


if __name__ == '__main__':
    img_dir = './samples/*'
    output_dir = './results/'
    
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path, 0) for img_path in images_paths])

    images = open_images(img_dir)

    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(tqdm(images)):
        results = fingerprint(img)
        cv.imwrite(output_dir + str(i) + '.png', results)


