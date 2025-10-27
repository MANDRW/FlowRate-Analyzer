from keras.mixed_precision import set_global_policy
import tensorflow as tf

import preparing
import model_builder
import coach
def main():
    set_global_policy('mixed_float16')
    gpus = tf.config.list_physical_devices('GPU')
    path = "all_rotate_dataset"
    img_size = (4032 // 10, 3024 // 10)
    batch_size = 16
    test_size = 0.3
    random_state = 42
    crop_ratio = 1
    name="newer_far_1"

    data_prep = preparing.DataPrep(path, img_size, batch_size, test_size, random_state, crop_ratio)
    train_gen, val_gen = data_prep.prep(augment=True, crop=False)

    builder= model_builder.ModelBuilder(input_shape=(img_size[0], img_size[1], 3), num_classes=3)
    model=builder.model()
    model.summary()

    coach_instance = coach.Coach(model, train_gen, val_gen, epochs=50, name=name)
    history = coach_instance.train(save=True)
    coach_instance.accuracy(history)
if __name__ == "__main__":
    main()