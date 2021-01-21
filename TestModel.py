from os import path
import sys
from MiniDeepFont import DeepFont
import Database
import Results

'''
Fonts crop size, after scaling. This is the size of the NN input.
'''
shape = (28, 28)

'''
Files names for testing
'''
test_filenames = [ 'datasets/test/test.h5' ]
font_test_db = 'test_fonts_temp_db.h5'

'''
Model file name
'''
model_filename_full = 'models/MiniDeepFont.model.h5'

'''
Results filename
'''
test_results_file = 'test_results.csv'

# python3 TestModel <model file name> <test set h5 file name> <csv result file name> <intermediate temp fil>
print(sys.argv)
if len(sys.argv) > 1:
    model_filename_full = sys.argv[1]
if len(sys.argv) > 2:
    test_filenames = [ sys.argv[2] ]
if len(sys.argv) > 3:
    test_results_file = sys.argv[3]
if len(sys.argv) > 4:
    font_test_db = sys.argv[4]

# Loading the model
deep_font = DeepFont(shape + (3,))
deep_font.load(model_filename_full)
print('Model loaded')

# Test preprocess & loading
Database.prepare(test_filenames, font_test_db, shape, store_labels=False, rewrite=True)
test_x, _, test_letters, test_filenames = Database.load(font_test_db, load_labels=False)

# Running model
predictions = deep_font.predict(test_x)

# Saving results
Results.store(test_results_file, predictions, test_filenames, test_letters)
