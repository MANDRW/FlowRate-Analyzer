from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.legacy_tf_layers.core import dropout


class ModelBuilder:
    def __init__(self, input_shape=(672,504,3),num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def model(self):
        model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=self.input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
        ])
        optimazer=Adam(learning_rate=0.00001)
        model.compile(optimizer=optimazer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model



