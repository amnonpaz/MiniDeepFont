from os import path
from MiniDeepFont import DeepFont
import matplotlib.pyplot as plt
import Database
import Results

'''
Fonts crop size, after scaling. This is the size of the NN input.
'''
shape = (28, 28)

'''
Files names for training and validation
If validation files list is empty, validation will not be executed
'''
train_filenames = [ 'datasets/train/SynthText.h5', 'datasets/train/train.h5' ]
font_train_db = 'datasets/train/ExtractedFonts.h5'
validation_filenames = [ 'datasets/validation/SynthTextValidation.h5' ]
font_validation_db = 'datasets/validation/ExtractedFontsValidation.h5'
validate_results_file = 'datasets/validation/results.csv'

'''
Model file name, without suffix
The actual file name will have the suffix '.h5'
'''
model_filename = 'models/MiniDeepFont.model'


# Training preprocess: Fonts are saved in a database for caching, then loaded
Database.prepare(train_filenames, font_train_db, shape, rewrite=True)
train_x, train_y, _, _ = Database.load(font_train_db)

# Initializing NN
deep_font = DeepFont(shape + (3,), opt_name='sgd', use_augmentations=True)

# Training only if the model file doesn't exist
model_filename_full = model_filename + '.h5'
if not path.exists(model_filename_full):
    deep_font.summarize()
    results = deep_font.train(train_x, train_y, 2, 32)
    print('Model Loss: {L} ; Accuracy: {A}'.format(L=results['evaluation'][0], A=results['evaluation'][1]))
    deep_font.save(model_filename)
else:
    # Loading the model
    deep_font.load(model_filename_full)
    print('Model loaded')

if len(validation_filenames) > 0:
    # Validation preprocess & loading
    Database.prepare(validation_filenames, font_validation_db, shape, rewrite=True)
    validate_x, validate_y, validate_letters, validate_filenames = Database.load(font_validation_db)

    # Executing evaluation
    predictions = deep_font.predict(validate_x)

    # Printing & saving results
    evaluation = deep_font.evaluate(validate_x, validate_y)
    print('Validation Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))
    Results.store(validate_results_file, predictions, validate_filenames, validate_letters)
