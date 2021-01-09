from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers


class DeepFont:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(48, 48), activation='relu', input_shape=input_shape, name='conv_1'))
        self.model.add(BatchNormalization(name='norm_1'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_1'))

        self.model.add(Conv2D(128, kernel_size=(24, 24), activation='relu', name='conv_2'))
        self.model.add(BatchNormalization(name='norm_2'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='pooling_2'))

        self.model.add(Conv2DTranspose(128, (24, 24), strides=(2, 2), activation='relu', padding='same',
                                       kernel_initializer='uniform'))
        self.model.add(UpSampling2D(size=(2, 2)))

        self.model.add(Conv2DTranspose(64, (12, 12), strides=(2, 2), activation='relu', padding='same',
                                       kernel_initializer='uniform'))
        self.model.add(UpSampling2D(size=(2, 2)))

        self.model.add(Conv2D(256, kernel_size=(12, 12), activation='relu', name='conv_3'))
        self.model.add(Conv2D(256, kernel_size=(12, 12), activation='relu', name='conv_4'))
        self.model.add(Conv2D(256, kernel_size=(12, 12), activation='relu', name='conv_5'))

        self.model.add(Flatten())

        self.model.add(Dense(4096, activation='relu', name='fc6'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu', name='fc7'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2383, activation='relu', name='fc8'))

        self.model.add(Dense(1, activation='softmax', name='softmax_classifier'))

        opt = optimizers.SGD(lr=0.01)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model.summary()

    def train(self, images, labels, epochs, batch_size):
        self.model.fit(images, labels, epochs=epochs, batch_size=batch_size)
        return self.model.evaluate(images, labels)

    def evaluate(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels, verbose=0)

    def save(self, filename):
        self.model.save(filename + '.h5')

    def load(self, filename):
        self.model = load_model(filename)
