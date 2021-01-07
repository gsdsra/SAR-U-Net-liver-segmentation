import numpy as np
import cv2
from skimage.exposure import equalize_hist
from skimage.restoration import denoise_wavelet
import os
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    # print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def elastic(image, alpha, sigma):
    random_stae = np.random.RandomState(42)
    elastic_image = elastic_transform(image, alpha, sigma, random_stae)
    return elastic_image


def zoom(image, factor, fill_color):
    size = np.shape(image)
    slice_image = Image.fromarray(image)
    new_hight = size[0]*factor
    new_width = size[1]*factor

    center = [size[0]/2, size[1]/2]
    x1 = int(center[0] - new_width/2)
    y1 = int(center[1] - new_hight/2)
    x2 = int(center[0] + new_width/2)
    y2 = int(center[1] + new_hight/2)
    zoomed_image = np.array(slice_image.transform((size[0], size[1]), Image.EXTENT, [x1, y1, x2, y2], fillcolor=fill_color))
    return zoomed_image


def rotate_image(image, fill_color):
    num = np.random.randint(0, 15, 1)
    size = np.shape(image)
    slice_image = Image.fromarray(image)
    rotated_image = slice_image.rotate(num, fillcolor=fill_color)
    return rotated_image


def flip(image, axis):
    if axis == 0:
        zoomed_image = np.fliplr(image)
    elif axis == 1:
        zoomed_image = np.flipud(image)
    return zoomed_image


def normalize_data(data):
    data[data < -200] = -200
    data[data > 255] = 255
    data_1 = equalize_hist(data)
    data_1 = denoise_wavelet(data_1)
    return data_1
#
# #直方图均衡化
# def histeq(img_path, nbr_bins=256):
#     img = Image.open(img_path)
#     img = np.array(img, "f")
#     # """ Histogram equalization of a grayscale image. """
#     imhist, bins, patches = plt.hist(img.flatten(), nbr_bins, normed=True)
#
#     cdf = imhist.cumsum()  # cumulative distribution function
#     cdf = 255 * cdf / cdf[-1]
#
#     result = np.interp(img.flatten(), bins[:-1], cdf)
#     result = result.reshape(img.shape)
#     imageio.imwrite(img_path.replace("test_corre2d", "test_corre2d_histeq"), result)


