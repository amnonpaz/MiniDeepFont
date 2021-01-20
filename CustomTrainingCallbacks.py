from keras.callbacks import Callback

class EvaluteTrainSet(Callback):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        self.history = {
                'training_accuracy': [],
                'training_loss': []
                }


    def on_epoch_end(self, epoch, logs=None):
        evaluation = self.model.evaluate(self.train_x, self.train_y, verbose=0)
        self.history['training_loss'].append(evaluation[0])
        self.history['training_accuracy'].append(evaluation[1])
        print(' ; Train set: Loss: {L} , Accuracy: {A}'.format(L=evaluation[0], A=evaluation[1]))


    def get_history(self):
        return self.history
