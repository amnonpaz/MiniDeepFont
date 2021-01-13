import cv2
from skimage import util, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

class CustomAugmentations(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(preprocessing_function=self.augment, **kwargs)

        random.seed()
                
    def augment(self, image):
        if (random.randint(0, 1) == 0):
            return image

        #return self.add_shading(self.add_rotation(self.add_blur(self.add_noise(image))))
        return self.add_rotation(self.add_blur(self.add_noise(image)))

    def add_noise(self, image):
        if (random.randint(0, 1) == 0):
            return image

        return util.random_noise(image, mode='gaussian', clip=True, mean=0, var=3)

    def add_blur(self, image):
        if (random.randint(0, 1) == 0):
            return image

        kernel_size = 5
        sigma = random.uniform(2.5, 3.5)
        return cv2.GaussianBlur(image ,(kernel_size, kernel_size), sigma)

    def add_rotation(self, image):
        if (random.randint(0, 1) == 0):
            return image

        angle = random.uniform(-10, 10)
        return transform.rotate(image, angle)

    def add_shading(self, image):
        if (random.randint(0, 1) == 0):
            return image

        return image + cv2.Laplacian(image, cv2.CV_64F).astype('float32')
