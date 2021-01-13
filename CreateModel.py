import h5py
import numpy as np
import Preprocess
from os import path
from MiniDeepFont import DeepFont
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



def db_add_datadet(db_group, img, label,key: str, bb_idx: str, postfix=''):
    dataset = db_group.create_dataset('{K}_{I}_{P}'.format(K=key, I=bb_idx, P=postfix),
                                      shape=img.shape, data=img, dtype='f')
    dataset.attrs['label'] = label


def prepare_database(hdf5_input: str, hdf5_output: str, shape: tuple,
                     affine_crop=False, do_norm=True, rewrite=False, augment=False):

    if augment:
        augmentations = {
                0: [ Preprocess.unit, Preprocess.add_noise ],
                1: [ Preprocess.unit ],
                2: [ Preprocess.unit, Preprocess.add_noise, Preprocess.rotate]
                }
    else:
        augmentations = {
                0: [ Preprocess.unit ],
                1: [ Preprocess.unit ],
                2: [ Preprocess.unit ]
                }

    with h5py.File(hdf5_input, 'r') as images_db:
        if not rewrite and path.exists(hdf5_output):
            return

        with h5py.File(hdf5_output, 'w') as fonts_db:
            images_group = fonts_db.create_group('images')

            print('Processing file {F}'.format(F=hdf5_input))

            keys = list(images_db['data'].keys())
            for key in keys:
                img = images_db['data'][key][:]
                if do_norm:
                    img = Preprocess.normalize_image(img)

                bboxes = images_db['data'][key].attrs['charBB']
                fonts = images_db['data'][key].attrs['font']

                for bb_idx in range(bboxes.shape[-1]):
                    bb = bboxes[:, :, bb_idx]
                    character = Preprocess.crop_character(img, bb, shape, affine_crop)
                    label = encode_font_name(fonts[bb_idx].decode('utf-8'))
                    #db_add_datadet(images_group, character, label, key, bb_idx, 'clean')

                    for func in augmentations[label]:
                        aug = func(character)
                        db_add_datadet(images_group, aug, label, key, bb_idx, func.__name__)


def load_database(filename: str):
    with h5py.File(filename, 'r') as db:
        keys = list(db['images'].keys())
        images = np.array([np.array(db['images'][k][:]) for k in keys])
        labels = np.array([db['images'][k].attrs['label'] for k in keys])

        print('Total number of lables: {L}'.format(L=labels.shape[0]))
        for l in range(labels.max() + 1):
            print('Total number lables #{I}: {L}'.format(I=l, L=np.count_nonzero(labels == l)))

        return images, labels


if __name__ == '__main__':
    input_filename = 'datasets/train/SynthText.h5'
    validation_filename = 'datasets/validation/SynthTextValidation.h5'
    font_train_db = 'datasets/train/ExtractedFonts.h5'
    font_validation_db = 'datasets/validation/ExtractedFontsValidation.h5'

    shape = (28, 28)

    prepare_database(input_filename, font_train_db, shape, rewrite=True, augment=False)
    prepare_database(validation_filename, font_validation_db, shape, rewrite=False)

    train_x, train_y = load_database(font_train_db)
    validate_x, validate_y = load_database(font_validation_db)

    model_filename = 'models/MiniDeepFont.model'
    deep_font = DeepFont(shape + (3,), opt_name='sgd')
    if not path.exists(model_filename):
        evaluation = deep_font.train(train_x, train_y, 20, 32)
        print('Model Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))
        deep_font.save(model_filename)
    else:
        deep_font.load(model_filename)
        print('Model loaded')

    evaluation = deep_font.evaluate(validate_x, validate_y)
    print('Test Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))
 
