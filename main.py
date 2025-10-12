import preparing
import model_builder
import coach
def main():
    path = "close_flowrate_dataset"
    img_size = (4032 // 8, 3024 // 8)
    batch_size = 16
    test_size = 0.2
    random_state = 42
    crop_ratio = 0.99
    name="newer_close_1"

    data_prep = preparing.DataPrep(path, img_size, batch_size, test_size, random_state, crop_ratio)
    train_gen, val_gen = data_prep.prep(augment=True, crop=True)

    builder= model_builder.ModelBuilder(input_shape=(img_size[0], img_size[1], 3), num_classes=3)
    model=builder.model()
    model.summary()

    coach_instance = coach.Coach(model, train_gen, val_gen, epochs=100, name=name)
    history = coach_instance.train(save=True)
    coach_instance.accuracy(history)

if __name__ == "__main__":
    main()