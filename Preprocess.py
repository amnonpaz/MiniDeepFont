import cv2
import numpy as np
from skimage import util, transform



'''
Normalize the mean and std of each channel in an image
'''
def normalize_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype('float32')

#    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype('float32')
#    return image / 255
#    mean, std = image.mean(axis=(0, 1)), image.std(axis=(0, 1))
#    return (image - mean) / std / 255


'''
Extract one font crop from an image and reshapes it to a rectangle
image: Source image
points: np float32 array [ (x0,y0), (x1,y1), (x2, y2), (x3, y3) ]
result_shape: Rectangle's parameters - tuple (W, H)
'''
def extract_font_crop_from_image_affine(image, points, result_shape):
    dest_points = np.float32([[0, 0], [result_shape[0], 0],
                             [result_shape[0], result_shape[1]], [0, result_shape[1]]])
    points = points[:3]
    dest_points = dest_points[:3]
    perspective = cv2.getAffineTransform(points, dest_points)
    return cv2.warpAffine(image, perspective, result_shape)


def extract_font_crop_from_image_perspective(image, points, result_shape):
    dest_points = np.float32([[0, 0], [result_shape[0], 0],
                           [result_shape[0], result_shape[1]], [0, result_shape[1]]])

    perspective = cv2.getPerspectiveTransform(points, dest_points)
    return cv2.warpPerspective(image, perspective, result_shape)


def crop_character(image, bb, shape, affine=False):
    if not affine:
        points = np.float32([bb[0, :], bb[1, :]]).T
        character = extract_font_crop_from_image_perspective(image, points, shape)
    else:
        points = np.float32([bb[0, :3], bb[1, :3]]).T
        character = extract_font_crop_from_image_affine(image, points, shape)

    return character


def unit(image):
    return image


def blur(image):
    kernel_size = 3
    return cv2.GaussianBlur(image ,(kernel_size, kernel_size), 0)


def add_noise(image):
    return util.random_noise(image, mode='gaussian', clip=True)


def rotate(image):
    return transform.rotate(image, 9)
