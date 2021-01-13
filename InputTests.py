import matplotlib.pyplot as plt
import cv2
import h5py
import Preprocess
from math import floor

def input_test(hdf5_filename: str, sample_index: int, shape: tuple, affine_crop=False, do_norm=True, show_input=False):

    with h5py.File(hdf5_filename, 'r') as images_db:
        imgs_keys = list(images_db['data'].keys())
        key = imgs_keys[sample_index]
        img = images_db['data'][key][:]
        
        '''
        for attr in images_db['data'][key].attrs:
            print('Available attribure {A}: {V}'.format(A=attr, V=images_db['data'][key].attrs[attr]))
        '''

        bboxes = images_db['data'][key].attrs['charBB']
        fonts = images_db['data'][key].attrs['font']
        letters = ''.join([word.decode('utf-8') for word in images_db['data'][key].attrs['txt']])
        print('Letters: {L}'.format(L=letters))

        # If we normalize, we want to see the difference between
        # the original image and the normalized image
        if show_input and do_norm:
            plt.figure('Test image original')
            plt.imshow(img)
            plt.show(block=False), plt.xticks([]), plt.yticks([])

            img = Preprocess.normalize_image(img)
            plt.figure('Test image normalized')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_YUV2RGB))
            plt.show(block=False), plt.xticks([]), plt.yticks([])

        plt.figure('Fonts')
        plt.subplot()
        fonts_rows = 3
        fonts_to_display = min(bboxes.shape[-1], fonts_rows*3)
        for bb_idx in range(fonts_to_display):
            bb = bboxes[:, :, bb_idx]
            character = Preprocess.crop_character(img, bb, shape, affine_crop)

            plt.subplot(fonts_rows, floor(fonts_to_display/fonts_rows), bb_idx + 1)
            plt.imshow(character if not do_norm else cv2.cvtColor(character, cv2.COLOR_YUV2RGB))
            plt.title(letters[bb_idx] + ': ' + fonts[bb_idx].decode("utf-8")), plt.xticks([]), plt.yticks([])

        plt.show()


if __name__ == '__main__':
    input_filename = 'datasets/train/SynthText.h5'
    validation_filename = 'datasets/validation/SynthTextValidation.h5'
    font_train_db = 'datasets/train/ExtractedFonts.h5'
    font_validation_db = 'datasets/validation/ExtractedFontsValidation.h5'

    shape = (28, 28)
    input_test(input_filename , 450, shape)
