from keras.callbacks import Callback

class EvaluteTrainSet(Callback):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def on_epoch_end(self, epoch, logs=None):
        evaluation = self.model.evaluate(self.train_x, self.train_y, verbose=0)
        print(' ; Train set: Loss: {L} , Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))
