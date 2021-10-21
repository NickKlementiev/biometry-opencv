import numpy as np
import scipy

def gabor_filter(img, orient, freq, kx=0.65, ky=0.65):
    angle_inc = 3
    img = np.double(img)
    rows, cols = img.shape
    return_img = np.zeros((rows, cols))

    freq_1d = freq.flatten()
    frequency_ind = np.array(np.where(freq_1d > 0))
    non_zero_elems = freq_1d[frequency_ind]
    non_zero_elems = np.double(np.round((non_zero_elems * 100))) / 100
    unfreq = np.unique(non_zero_elems)

    sigma_x = 1 / unfreq * kx
    sigma_y = 1 / unfreq * ky
    block_size = int(np.round(3 * np.max([sigma_x, sigma_y])))
    array = np.linspace(-block_size, block_size, (2 * block_size + 1))
    x, y = np.meshgrid(array, array)

    reffilter = np.exp(-(((np.power(x,2))/(sigma_x*sigma_x) + (np.power(y,2))/(sigma_y*sigma_y)))) * np.cos(2*np.pi*unfreq[0]*x)
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180 // angle_inc, filt_rows, filt_cols)))

    for degree in range(0, 180 // angle_inc):
        rot_filt = scipy.ndimage.rotate(reffilter, - (degree * angle_inc + 90), reshape=False)
        gabor_filter[degree] = rot_filt

    maxorientindex = np.round(180 / angle_inc)
    orientindex = np.round(orient / np.pi * 180 / angle_inc)

    for i in range(0, rows // 16):
        for j in range(0, cols // 16):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex
                
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq > 0)
    finalind = np.where((valid_row > block_size) &
            (valid_row < rows - block_size) &
            (valid_col > block_size) &
            (valid_col < cols - block_size))

    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]]
        c = valid_col[finalind[0][k]]
        img_block = img[r - block_size: r + block_size + 1][:, c-block_size: c + block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1])

    gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

    return gabor_img

