import h5py
import numpy as np
import Preprocess
from os import path
from MiniDeepFont import DeepFont
import matplotlib.pyplot as plt
import csv

fonts_attrs = {
    'Skylark': {
        'color': 'r',
        'code': 0
    },
    'Sweet Puppy': {
        'color': 'b',
        'code': 1
    },
    'Ubuntu Mono': {
        'color': 'g',
        'code': 2
    }
}


def encode_font_name(font_name: str):
    return fonts_attrs[font_name]['code']


def decode_font_name(font_code: int):
    for k in fonts_attrs:
        if fonts_attrs[k]['code'] == font_code:
            return k


def fonts_list():
    return list(fonts_attrs.keys())


def bb_color(font):
    tmp = font.decode('UTF-8')
    res = 'b'
    for font_name, data in fonts_attrs.items():
        if tmp == font_name:
            res = data['color']
            break
    return res


def db_add_datadet(db_group, img, label, letter, filename: str, bb_idx: str, postfix=''):
    dataset = db_group.create_dataset('{F}_{I}_{P}'.format(F=filename, I=bb_idx, P=postfix),
                                      shape=img.shape, data=img, dtype='f')
    dataset.attrs['label'] = label
    dataset.attrs['filename'] = filename
    dataset.attrs['letter'] = letter


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
                letters = ''.join([word.decode('utf-8') for word in images_db['data'][key].attrs['txt']])

                for bb_idx in range(bboxes.shape[-1]):
                    bb = bboxes[:, :, bb_idx]
                    character = Preprocess.crop_character(img, bb, shape, affine_crop)
                    label = encode_font_name(fonts[bb_idx].decode('utf-8'))

                    for func in augmentations[label]:
                        aug = func(character)
                        db_add_datadet(images_group, aug, label, letters[bb_idx], key, bb_idx, func.__name__)


def load_database(filename: str):
    with h5py.File(filename, 'r') as db:
        keys = list(db['images'].keys())
        images = np.array([np.array(db['images'][k][:]) for k in keys])
        labels = np.array([db['images'][k].attrs['label'] for k in keys])
        letters = np.array([db['images'][k].attrs['letter'] for k in keys])
        filenames = np.array([db['images'][k].attrs['filename'] for k in keys])

        print('Total number of lables: {L}'.format(L=labels.shape[0]))
        for l in range(labels.max() + 1):
            print('Total number lables #{I} ({S}): {L}'
                    .format(I=l, S=decode_font_name(l), L=np.count_nonzero(labels == l)))

        return images, labels, letters, filenames


def store_results(dest_filename, predictions, filenames, letters):
    with open(dest_filename, 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',')
        reswriter.writerow(['', 'image','char'] + fonts_list())
        idx = 0
        for v in np.argmax(predictions, axis=1):
            res = np.zeros(len(fonts_attrs))
            res[v] = 1.0
            reswriter.writerow([idx] + [filenames[idx], letters[idx]] + res.tolist())
            idx += 1


if __name__ == '__main__':
    input_filename = 'datasets/train/SynthText.h5'
    validation_filename = 'datasets/validation/SynthTextValidation.h5'
    font_train_db = 'datasets/train/ExtractedFonts.h5'
    font_validation_db = 'datasets/validation/ExtractedFontsValidation.h5'

    shape = (28, 28)

    prepare_database(input_filename, font_train_db, shape, rewrite=True, augment=False)
    prepare_database(validation_filename, font_validation_db, shape, rewrite=False)

    train_x, train_y, _, _ = load_database(font_train_db)
    validate_x, validate_y, validate_letters, validate_filenames = load_database(font_validation_db)

    model_filename = 'models/MiniDeepFont.model'
    model_filename_full = model_filename + '.h5'

    deep_font = DeepFont(shape + (3,), opt_name='sgd', use_augmentations=True)

    if not path.exists(model_filename_full):
        deep_font.summarize()
        results = deep_font.train(train_x, train_y, 50, 32)
        print('Model Loss: {L} ; Accuracy: {A}'.format(L=results['evaluation'][0], A=results['evaluation'][1]))
        deep_font.save(model_filename)
        for x in results['history']:
            plt.plot(results['history'][x], label=x)
        plt.legend()
        plt.show()
    else:
        deep_font.load(model_filename_full)
        print('Model loaded')
        deep_font.summarize()

    evaluation = deep_font.evaluate(validate_x, validate_y)
    print('Test Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))

    predictions = deep_font.predict(validate_x)
    store_results('datasets/validation/results.csv', predictions, validate_filenames, validate_letters)
