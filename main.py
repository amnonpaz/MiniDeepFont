import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import Preprocess
from os import path
from DeepFont import DeepFont
from sklearn.model_selection import train_test_split

fonts_attrs = {
    'Skylark': {
        'color': 'r',
        'code': 0
    },
    'Ubuntu Mono': {
        'color': 'g',
        'code': 1
    },
    'Sweet Puppy': {
        'color': 'b',
        'code': 2
    }
}


def encode_font_name(font_name: str):
    return fonts_attrs[font_name]['code']


def decode_font_name(font_code: int):
    for k in fonts_attrs:
        if fonts_attrs[k]['code'] == font_code:
            return k


def bb_color(font):
    tmp = font.decode('UTF-8')
    res = 'b'
    for font_name, data in fonts_attrs.items():
        if tmp == font_name:
            res = data['color']
            break
    return res


def input_test(hdf5_filename, sample_index, affine_crop=False, do_norm=True, show_input=False):
    with h5py.File(hdf5_filename, 'r') as images_db:
        imgs_keys = list(images_db['data'].keys())
        key = imgs_keys[sample_index]
        img = images_db['data'][key][:]

        bboxes = images_db['data'][key].attrs['charBB']
        fonts = images_db['data'][key].attrs['font']

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
        for bb_idx in range(bboxes.shape[-1]):
            bb = bboxes[:, :, bb_idx]
            character = Preprocess.crop_character(img, bb, (105, 105), affine_crop)

            plt.subplot(4, int(bboxes.shape[-1]/4 + 1), bb_idx + 1)
            plt.imshow(character if not do_norm else cv2.cvtColor(character, cv2.COLOR_YUV2RGB))
            plt.title(fonts[bb_idx].decode("utf-8")), plt.xticks([]), plt.yticks([])

        plt.show()


def prepare_database(hdf5_input: str, hdf5_output: str, shape: tuple,
                     affine_crop=False, do_norm=True, rewrite=False):
    with h5py.File(hdf5_input, 'r') as images_db:
        if not rewrite and path.exists(hdf5_output):
            return

        with h5py.File(hdf5_output, 'w') as fonts_db:
            images_group = fonts_db.create_group('images')

            keys = list(images_db['data'].keys())
            for key in keys:
                print('Processing image {I}'.format(I=key))
                img = images_db['data'][key][:]
                if do_norm:
                    img = Preprocess.normalize_image(img)

                bboxes = images_db['data'][key].attrs['charBB']
                fonts = images_db['data'][key].attrs['font']

                for bb_idx in range(bboxes.shape[-1]):
                    bb = bboxes[:, :, bb_idx]
                    character = Preprocess.crop_character(img, bb, shape, affine_crop)
                    dataset = images_group.create_dataset('{K}_{I}'.format(K=key, I=bb_idx), shape=character.shape, data=character, dtype='f')
                    dataset.attrs['label'] = encode_font_name(fonts[bb_idx].decode('utf-8'))


def load_database(filename: str):
    with h5py.File(filename, 'r') as db:
        keys = list(db['images'].keys())
        images = np.array([db['images'][k][:] for k in keys])
        labels = np.array([db['images'][k].attrs['label'] for k in keys])

        return images, labels


if __name__ == '__main__':
    input_filename = 'datasets/initial/SynthText.h5'
    font_images_filename = 'datasets/initial/ExtractedFonts.h5'
    shape = (105, 105)
    prepare_database(input_filename, font_images_filename, shape, rewrite=False)

    images, labels = load_database(font_images_filename)
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.25)

    model_filename = 'DeepFont.model'
    deep_font = DeepFont(shape + (3,))
    if not path.exists(model_filename):
        evaluation = deep_font.train(train_x, train_y, 2, 128)
        print('Model evaluation: {E}'.format(E=evaluation))
    else:
        deep_font.load(model_filename)
        print('Model loaded')
 