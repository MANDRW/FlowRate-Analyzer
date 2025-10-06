import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.version_utils import callbacks


class Coach:
    def __init__(self, model, train_data, val_data, epochs=15,name="model"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.name = name

    def train(self,save=False):
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.epochs,
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        if(save):
            self.model.save("models/"+self.name+".h5")
        return history

    def accuracy(self, history):
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()