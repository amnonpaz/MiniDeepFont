import h5py
import numpy as np
import Preprocess
import Fonts


def add_dataset(db_group, img, letter, filename: str, idx: str, label = None):
    dataset = db_group.create_dataset('{F}_{I:04d}'.format(F=filename, I=idx),
                                      shape=img.shape, data=img, dtype='float32')
    dataset.attrs['filename'] = filename
    dataset.attrs['letter'] = letter
    if label is not None:
        dataset.attrs['label'] = label


def prepare(input_files: list, hdf5_output: str, shape: tuple,
            affine_crop=False, rewrite=False, store_labels=True):

    for hdf5_input in input_files:
        with h5py.File(hdf5_input, 'r') as images_db:
            if not rewrite and path.exists(hdf5_output):
                return

            with h5py.File(hdf5_output, 'w') as fonts_db:
                images_group = fonts_db.create_group('images')

                print('Processing file {F}'.format(F=hdf5_input))

                for key in images_db['data'].keys():
                    img = Preprocess.normalize_image(images_db['data'][key][:])

                    bboxes = images_db['data'][key].attrs['charBB']
                    fonts = images_db['data'][key].attrs['font']
                    letters = ''.join([word.decode('utf-8') for word in images_db['data'][key].attrs['txt']])

                    for bb_idx in range(bboxes.shape[-1]):
                        bb = bboxes[:, :, bb_idx]
                        character = Preprocess.crop_character(img, bb, shape, affine_crop)
                        label = Fonts.encode_name(fonts[bb_idx].decode('utf-8')) if store_labels else None

                        add_dataset(images_group, character, letters[bb_idx], key, bb_idx, label)


def load(filename: str, load_labels = True):
    with h5py.File(filename, 'r') as db:
        keys = db['images'].keys()
        images = np.array([np.array(db['images'][k][:]) for k in keys])
        letters = np.array([db['images'][k].attrs['letter'] for k in keys])
        filenames = np.array([db['images'][k].attrs['filename'] for k in keys])
        if load_labels:
            labels = np.array([db['images'][k].attrs['label'] for k in keys])

        print('Total number of lables: {L}'.format(L=labels.shape[0]))
        for l in range(labels.max() + 1):
            print('Total number lables #{I} ({S}): {L}'
                    .format(I=l, S=Fonts.decode_name(l), L=np.count_nonzero(labels == l)))

        return images, labels, letters, filenames
