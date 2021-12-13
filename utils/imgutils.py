
import os
import cv2
import sys
import math
import random
import numpy as np


def get_new_shape(images: np.ndarray, size: tuple(int, int) = None, n: int = None) -> tuple:
    """

    Create shape from specified size or number of images for images array. 

    :param images: image data in numpy array
        :shape: (n_init, h_init, w_init, c_init)

    :param size  : new size for images in tuple
        :example: (h_new, w_new)
    
    :param n     : number of images in new shape

    :return: shape of images array with specified size.
        :example: (n, h_new, w_new)

    """
    
    shape = list(images.shape)

    if size is not None:

        h, w = size

        shape[1] = h
        shape[2] = w

    if n is not None:

        shape[0] = n

    shape = tuple(shape)

    return shape


def random_crop(images: np.ndarray, size: tuple(int, int)) -> np.ndarray:
    """
    
    Crop images with random size.

    :param images: image data in numpy array
        :shape: (n_init, h_init, w_init, c_init)

    :param size  : crop size
        :shape: (h_c, w_c)

    :return: cropped image data in numpy array
        :shape: (n_init, h_c, w_c)

    """

    n_init, h_init, w_init = images.shape[:3]

    h_crop, w_crop = size

    shape_after_crop = get_new_shape(images, size)

    assert (h_crop >= h_init and w_crop >= w_init), "Crop size must be smaller than initial size!"

    cropped_images = np.ndarray(shape_after_crop, dtype=images.dtype)

    # get random crop size
    x = np.random.randint(low = 0, high = h_init - h_crop + 1, size = (n_init))
    y = np.random.randint(low = 0, high = w_init - w_crop + 1, size = (n_init))

    for i in range(n_init):
        cropped_images[i] = images[i, y[i]:y[i] + h_crop, x[i]:x[i] + w_crop]

    return cropped_images


def center_crop(images: np.ndarray, size: tuple(int, int)) -> np.ndarray:
    """
    
    Center crop all images.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param size  : crop size
        :shape: (h_c, w_c)

    :return: cropped image data in numpy array
        :shape: (n, h_c, w_c)
    
    """

    n, h_init, w_init = images.shape[:3]

    h_c, w_c = size

    assert (h_c > h_init or w_c > w_init)

    x = int(round(0.5 * (w_init - w_c)))
    y = int(round(0.5 * (h_init - h_c)))

    cropped = images[:, y:y+h_c, x:x+w_c]

    return cropped


def random_flip(images: np.ndarray) -> np.ndarray:
    """

    Random flip some of images.

    :param images: image data in numpy array
        :shape: (n, h, w, c)
    
    :return: flipped image data in numpy array
        :shape: (n, h, w, c)

    """

    n = images.shape[0]

    flipped = images.copy()

    # image indices will be flipped
    flips = np.random.rand(n) >= 0.5

    for i in range(n):
        if flips[i]:
            flipped[i] = np.fliplr(images[i])

    return flipped


def flip(images: np.ndarray) -> np.ndarray:
    """

    Flip all of images.

    :param images: image data in numpy array
        :shape: (n, h, w, c)
    
    :return: flipped image data in numpy array
        :shape: (n, h, w, c)

    """

    n = images.shape[0]

    flipped = images.copy()

    for i in range(n):
        flipped[i] = np.fliplr(images[i])

    return flipped


def resize(images: np.ndarray, size: tuple(int, int)) -> np.ndarray:
    """
    
    Resize images with to size.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param size  : new size
        :shape: (h_n, w_n)

    :return: resized image data in numpy array
        :shape: (n, h_n, w_n, c_init)
    
    """

    n, h_init, w_init = images.shape[:3]

    h_n, w_n = size

    resized_shape = get_new_shape(images=images, size=size)

    resized = np.ndarray(resized_shape, dtype=images.dtype)

    for i in range(n):
        resized[i] = cv2.resize(images[i], dsize=(h_n, w_n))

    return resized


def padding(images: np.ndarray, padding: tuple) -> np.ndarray:
    """

    Pad images with the padding size.

    :param images   : image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param padding  : padding size for top, bottom, left, right
        :shape: (pad_x, pad_y) or (pad_t, pad_b, pad_l, pad_r)

    :return: resized image data in numpy array
        :shape: (n, h_n, w_n, c_init)

    """

    n, h_init, w_init = images.shape[:3]

    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]

    else:
        pad_t, pad_b, pad_l, pad_r = padding

    padded_shape = (h_init + pad_t + pad_b, w_init + pad_l + pad_r)
    padded_shape = get_new_shape(images=images, size=padded_shape)

    padded_images = np.zeros(padded_shape, dtype=images.dtype)
    padded_images[:, pad_t:pad_t+h_init, pad_l:pad_l+h_init] = images

    return padded_images


def standardize_images(images: np.ndarray, standard: str):
    """

    Standardize images.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)


    :param standard: str standardization type
        :options: 
            - mean_scale
            - scale

    :return: numpy array of standardized images

    """

    if standard == "mean_scale":
        mean = 127.5
        std  = 128.0

    elif standard == "scale":
        mean = 0.0
        std  = 255.0
    
    else:
        raise ValueError("'scale' must be 'mean_scale' or 'scale'!")

    standardized_images = images.astype(np.float32)
    standardized_images -= mean
    standardized_images /= std

    return standardized_images


def random_shift(images: np.ndarray, max_ratio: float):
    """

    Random shift images.

    :param images   : image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param max_ratio: maximum ratio for shift

    :return: shifted images

    """

    n, h_init, w_init = images.shape[:3]

    pad_x = int(w_init * max_ratio) + 1
    pad_y = int(h_init * max_ratio) + 1

    padded_images = padding(images=images, padding=(pad_x, pad_y))
    shidted_images = images.copy()

    shift_x = (w_init * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (h_init * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        shidted_images[i] = padded_images[i, pad_y+shift_y[i]:pad_y+shift_y[i]+h_init,
                                            pad_x+shift_x[i]:pad_x+shift_x[i]+w_init]

    return shidted_images


def random_downsample(images: np.ndarray, min_ratio: float):
    """
    
    Downsample images.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :min_ratio   : minimum ratio for downsampling

    :return      : downsampled images
    
    """

    n, h_init, w_init = images.shape[:3]

    downsampled_images = images.copy()

    ratios = min_ratio + (1 - min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * w_init))
        h = int(round(ratios[i] * h_init))

        downsampled_images[i, :h, :w] = cv2.resize(images[i], size=(h, w))
        downsampled_images[i] = cv2.resize(downsampled_images[i, :h, :w], size=(h_init, w_init))

    return downsampled_images


def expand_flip(images: np.ndarray):
    """
    
    Expand each image and insert after the original image.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :return: image data in numpy array with flipped ones
        :shape: (2n, h_init, w_init, c_init)


    """

    n_init, h_init, w_init = images.shape[:3]

    expanded_shape = get_new_shape(images, n=2*n_init)

    expanded_images = np.stack([images, flip(images)], axis=1)
    expanded_images = expanded_images.reshape(expanded_shape)

    return expanded_images


def five_crop(images: np.ndarray, size: tuple):
    """
    
    Crops image from; top, bottom, left, right and center.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param size: size of cropped image
        :shape: (h_crop, w_crop)

    return: cropped images
        :shape: (5n, h_crop, w_crop, c_init)
    
    """

    n_init, h_init, w_init = images.shape[:3]
    h_crop, w_crop = size

    assert h_init < h_crop or w_init < w_crop, "Crop size must be smaller than image size!"

    cropped_images_shape = get_new_shape(images=images, size=size, n=5*n_init)
    
    cropped_images_list = []

    cropped_images_list.append(images[:,:h_crop,:w_crop])
    cropped_images_list.append(images[:,:h_crop,-w_crop:])
    cropped_images_list.append(images[:,-h_crop:,:w_crop])
    cropped_images_list.append(images[:,-h_crop:,-w_crop:])
    cropped_images_list.append(center_crop(images, size))

    cropped_images = np.stack(cropped_images_list, axis=1).reshape(cropped_images_shape)

    return cropped_images


def ten_crop(images: np.ndarray, size: tuple):
    """

    Five crops on original and five crops on flipped images.

    :param images: image data in numpy array
        :shape: (n, h_init, w_init, c_init)

    :param size: size of cropped image
        :shape: (h_crop, w_crop)

    return: cropped images
        :shape: (10n, h_crop, w_crop, c_init)

    """

    n_init, h_init, w_init = images.shape[:3]
    cropped_shape = get_new_shape(images=images, size=size, n=10*n_init)

    cropped_5 = five_crop(images=images, size=size)
    cropped_5_flipped = five_crop(images=flip(images), size=size)

    cropped_images = np.stack([cropped_5, cropped_5_flipped], axis=1)
    cropped_images = cropped_images.reshape(cropped_shape)

    return cropped_images