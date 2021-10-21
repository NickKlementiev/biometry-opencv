import numpy as np
import math
import scipy.ndimage

def frequest(img, orient_img, kernel_size, min_wave, max_wave):
    rows, cols = np.shape(img)

    cosorient = np.cos(2 * orient_img)
    sinorient = np.sin(2 * orient_img)
    block_orient = math.atan2(sinorient, cosorient) / 2

    rot_img = scipy.ndimage.rotate(img, block_orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

    cropsize = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsize) / 2))
    rot_img = rot_img[offset: offset + cropsize][:, offset: offset + cropsize]

    ridge_sum = np.sum(rot_img, axis=0)
    dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
    ridge_noise = np.abs(dilation - ridge_sum)
    peak_thresh = 2

    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)
    _, no_of_peaks = np.shape(maxind)

    if no_of_peaks < 2:
        freq_block = np.zeros(img.shape)
    else:
        wave_length = (maxind[0][-1] - maxind[0][0]) / (no_of_peaks - 1)

        if wave_length >= min_wave and wave_length <= max_wave:
            freq_block = 1 / np.double(wave_length) * np.ones(img.shape)
        else:
            freq_block = np.zeros(img.shape)

    return freq_block

def ridge_freq(img, mask, orient, block_size, kernel_size, min_wave, max_wave):
    rows, cols = img.shape
    freq = np.zeros((rows, cols))

    for row in range(0, rows - block_size, block_size):
        for col in range(0, cols - block_size, block_size):
            img_block = img[row: row + block_size][:, col: col + block_size]
            angle_block = orient[row // block_size][col // block_size]
            if angle_block:
                freq[row: row + block_size][:, col: col + block_size] = frequest(img_block, angle_block, kernel_size, min_wave, max_wave)

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)
    ind = np.array(ind)
    ind = ind[1, :]
    non_zero_elems = freq_1d[0][ind]
    medianfreq = np.median(non_zero_elems) * mask

    return medianfreq
