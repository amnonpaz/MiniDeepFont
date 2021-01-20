from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CustomAugmentations import CustomAugmentations
from CustomTrainingCallbacks import EvaluteTrainSet
import numpy as np


class DeepFont:
    def __init__(self, input_shape, opt_name='adam', loss='mean_squared_error', use_augmentations=False):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, name='conv_1'))
        self.model.add(BatchNormalization(name='norm_1'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_1'))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv_2'))
        self.model.add(BatchNormalization(name='norm_2'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_2'))

        self.model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
                       kernel_initializer='uniform', name='convT_3'))
        self.model.add(UpSampling2D(size=(2, 2)))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv_4'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv_5'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv_6'))

        self.model.add(Flatten())

        self.model.add(Dense(1024, activation='relu', name='fc_1'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu', name='fc_2'))

        self.model.add(Dense(3, activation='softmax', name='softmax_classifier'))

        lr = 0.01
        if opt_name == 'adam':
            opt = optimizers.Adam(lr=lr, epsilon=1.0)
        elif opt_name == 'sgd':
            opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy', metrics.AUC(curve="ROC")])

        self.datagen = CustomAugmentations() if use_augmentations else ImageDataGenerator()

    def summarize(self):
        self.model.summary()

    def train(self, images, labels, epochs, batch_size, validate_x = None, validate_y = None):

        # Preparing training set
        x = images
        y = utils.to_categorical(labels)
        self.datagen.fit(x)

        # Preparing validation set (if given)
        validation_data = None
        if validate_x is not None and validate_y is not None:
            validation_data=(validate_x, utils.to_categorical(validate_y))

        # Creating a callack which calculates the real training set
        # metrics at the end of each epoch
        train_set_evaluation_callback = EvaluteTrainSet(x, y)

        # Training
        history = self.model.fit(self.datagen.flow(x, y, batch_size=batch_size),
                                 steps_per_epoch=len(x) / batch_size,
                                 epochs=epochs,
                                 callbacks=[train_set_evaluation_callback],
                                 validation_data=validation_data)
        return { 'history': history.history,
                 'evaluation': self.model.evaluate(x, y, verbose=0) }

    def evaluate(self, test_images, test_labels):
        x = test_images
        y = utils.to_categorical(test_labels)
        return self.model.evaluate(x, y, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filename):
        self.model.save(filename + '.h5')

    def load(self, filename):
        self.model = load_model(filename)

    def save_plot(self, filename):
        utils.plot_model(self.model,
                         to_file=filename,
                         show_shapes=True,
                         show_layer_names=True,
                         expand_nested=False)
