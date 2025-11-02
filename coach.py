import json
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class Coach:
    def __init__(self, model, train_data, val_data, epochs=50,name="model"):
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
            callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6),
                       ModelCheckpoint(f"models/{self.name}.h5", monitor='val_loss', save_best_only=True)]
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


    def save_history_to_json(self):
        if self.history is None:
            return
        history_data = {
            "loss": [float(l) for l in self.history.history['loss']],
            "accuracy": [float(a) for a in self.history.history['accuracy']],
            "val_loss": [float(l) for l in self.history.history['val_loss']],
            "val_accuracy": [float(a) for a in self.history.history['val_accuracy']],
            "model_name": self.name
        }
        json_path = f"{self.name}.json"
        with open(json_path, "w") as f:
            json.dump(history_data, f, indent=4)

    @staticmethod
    def mean_accuracy(scores):
        return np.mean(scores), np.std(scores)

    def accuracy(self, history=None):
        if history is None:
            history = self.history
        if history is None:
            return
        acc = history.history['val_accuracy'][-1]
        print(f"Validation accuracy: {acc:.4f}")
        return acc
