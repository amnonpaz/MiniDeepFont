# MiniDeepFont: Fonts classification CNN

Based on DeepFont [Paper](https://arxiv.org/pdf/1507.03196v1.pdf) by Adobe

This is my final project in computer vision course #22982 of the Open University of Israel

The input file was generated from [SynthText](https://github.com/ankush-me/SynthText). The objective is to detect one out of three possible fonts. The network I created is based on the one Adobe suggests in their paper, minus some layers and the unsupervised learning stage, and it achieved ~95% accuracy.

When given .h5 file, the scripts create a cached fonts database, ready for training/predication.
The names of these files should be set by the user.

## Training
In CreateModel.py:
- Line 22: Set the list of training datasets
- Line 23: Set font cache database
- Line 24: Set the validation dataset file names (can be left empty for no validation)
- Line 25: Set the validation results filename
- Line 32: Set the model file name (.h5 suffix will be added)

Execution: python3 CreatModel.py

## Testing
Execute: 
python3 TestModel.py <model file name> <test set h5 file name> <csv result file name> <intermediate temp file>

- model file name: The model .h5 file
- test set h5 file name: Test set database .h5 file
- csv result file name: Test results file
- intermediate temp file: Fonts cache database

## Required packages
  * matplotlib
  * skimage 
  * tensorflow
  * keras
  * h5py
  * numpy
  * csv
