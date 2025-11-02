import json

from keras.mixed_precision import set_global_policy
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

import preparing
import model_builder
import coach

def main():
    set_global_policy('mixed_float16')
    path = "all_rotate_dataset"
    img_size = (224,224) #(4032 // 10, 3024 // 10)
    batch_size = 8
    test_size = 0.3
    random_state = 42
    crop_ratio = 1
    name_prefix = "mobilenet_texture_kfold"
    num_folds = 5
    epochs = 50

    data_prep = preparing.DataPrep(path, img_size, batch_size)
    folds_data = data_prep.get_folds(num_folds=num_folds, augment=True)

    all_scores = []

    for fold_no, (train_gen, val_gen) in enumerate(folds_data, 1):
        builder = model_builder.ModelBuilder(input_shape=(img_size[0], img_size[1], 3), num_classes=3)
        model = builder.texture_mobilenet()
        coach_instance = coach.Coach(model, train_gen, val_gen, epochs=epochs, name=f"{name_prefix}_fold{fold_no}")
        history = coach_instance.train(save=True)
        score = model.evaluate(val_gen)
        all_scores.append(score[1])
        coach_instance.save_history_to_json(history)

    mean_acc, std_acc = coach.Coach.mean_accuracy(all_scores)
    print(f"Średnia dokładność: {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()

