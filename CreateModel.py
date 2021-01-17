import numpy as np
from os import path
from MiniDeepFont import DeepFont
import matplotlib.pyplot as plt
import csv
import Database
import Fonts


def store_results(dest_filename, predictions, filenames, letters):
    with open(dest_filename, 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',')
        reswriter.writerow(['', 'image','char'] + Fonts.get_list())
        idx = 0
        for v in np.argmax(predictions, axis=1):
            res = np.zeros(Fonts.get_number())
            res[v] = 1.0
            reswriter.writerow([idx] + [filenames[idx], letters[idx]] + res.tolist())
            idx += 1


if __name__ == '__main__':
    input_filenames = [ 'datasets/train/SynthText.h5', 'datasets/train/train.h5' ]
    validation_filename = [ 'datasets/validation/SynthTextValidation.h5' ]
    font_train_db = 'datasets/train/ExtractedFonts.h5'
    font_validation_db = 'datasets/validation/ExtractedFontsValidation.h5'

    shape = (28, 28)

    Database.prepare(input_filenames, font_train_db, shape, rewrite=True)
    Database.prepare(validation_filename, font_validation_db, shape, rewrite=True)

    train_x, train_y, _, _ = Database.load(font_train_db)
    validate_x, validate_y, validate_letters, validate_filenames = Database.load(font_validation_db)

    model_filename = 'models/MiniDeepFont.model'
    model_filename_full = model_filename + '.h5'

    deep_font = DeepFont(shape + (3,), opt_name='sgd', use_augmentations=True)

    if not path.exists(model_filename_full):
        deep_font.summarize()
        results = deep_font.train(train_x, train_y, 2, 32)
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
    print('Validation Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))

    predictions = deep_font.predict(validate_x)
    store_results('datasets/validation/results.csv', predictions, validate_filenames, validate_letters)
