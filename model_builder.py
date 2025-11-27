from cv2.datasets import none
from keras import Model
from keras.applications import MobileNetV2, VGG16, VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.legacy_tf_layers.core import dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Activation, concatenate


class ModelBuilder:
    def __init__(self, input_shape=(672, 504, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def model(self):
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')
        ])

        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    def mobilenet(self):
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
            for layer in base.layers[:80]:
                layer.trainable = False

            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x)
            x = Dense(128, activation='relu')(x)
            output = Dense(3, activation='softmax')(x)

            model = Model(inputs=base.input, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model


