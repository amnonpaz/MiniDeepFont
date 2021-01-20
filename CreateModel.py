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
Training parameters
'''
epochs = 20
batch_size = 32

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


'''
MAIN
'''

# Training preprocess: Fonts are saved in a database for caching, then loaded
Database.prepare(train_filenames, font_train_db, shape, rewrite=True)
train_x, train_y, _, _ = Database.load(font_train_db)

# Loading validation set if defined
validate_x, validate_y = None, None
if len(validation_filenames) > 0:
    Database.prepare(validation_filenames, font_validation_db, shape, rewrite=True)
    validate_x, validate_y, validate_letters, validate_filenames = Database.load(font_validation_db)

# Initializing NN
deep_font = DeepFont(shape + (3,), opt_name='sgd', use_augmentations=True)
deep_font.summarize()

# Training
model_filename_full = model_filename + '.h5'
results = deep_font.train(train_x, train_y, epochs, batch_size, validate_x, validate_y)
print('Model Loss: {L} ; Accuracy: {A}'
      .format(L=results['evaluation'][0], A=results['evaluation'][1]))
deep_font.save(model_filename)

plt.figure('Training results: Accurracy & Loss')
for k in results['history']:
    #if k.find('accuracy') != -1:
    plt.plot(results['history'][k], label=k)
plt.legend()
plt.title('Training results: Accurracy & Loss')
plt.show()

if len(validation_filenames) > 0:
    # Executing evaluation
    predictions = deep_font.predict(validate_x)

    # Printing & saving results
    evaluation = deep_font.evaluate(validate_x, validate_y)
    print('Validation Loss: {L} ; Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))
    Results.store(validate_results_file, predictions, validate_filenames, validate_letters)
