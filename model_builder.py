from keras import Model
from keras.applications import MobileNetV2
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
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.6),
            Dense(3, activation='softmax')
        ])
        optimizer = Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def texture_mobilenet(input_shape=(224, 224, 3), num_classes=3):
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base.layers[:80]:
            layer.trainable = False

        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def texture_net(input_shape=(128, 128, 3), num_classes=3):
        inp = Input(shape=input_shape)
        x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
        x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(inp)
        x3 = Conv2D(32, (7, 7), activation='relu', padding='same')(inp)

        merged = concatenate([x1, x2, x3])
        pooled = GlobalAveragePooling2D()(merged)
        out = Dense(num_classes, activation='softmax')(pooled)

        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

