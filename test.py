import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


MODEL_PATH = "models/mobilenet_0_9258/mobilenet_texture_kfold_fold1.h5"
IMAGE_PATH = "1_100.png"
IMG_SIZE = (224, 224)

CLASS_NAMES = ["Under", "Good", "Over"]


def load_and_prepare_image(path, img_size):
    img = load_img(path, target_size=img_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def predict_and_show():
    print(f"Ładowanie modelu z: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    img_original, img = load_and_prepare_image(IMAGE_PATH, IMG_SIZE)

    pred = model.predict(img)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]

    predicted_label = CLASS_NAMES[class_id]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    plt.axis("off")
    plt.title(
        f"Predykcja: {predicted_label}\n"
        f"Pewność: {confidence*100:.2f}%\n"
        f"Nazwa: {IMAGE_PATH}   Raw: {np.round(pred, 4)}",
        fontsize=12
    )
    plt.show()

    print("\n===== WYNIK =====")
    print(f"Obraz: {IMAGE_PATH}")
    print(f"Klasa: {predicted_label}")
    print(f"Raw: {np.round(pred, 4)}")
    print(f"Pewność: {confidence*100:.2f}%")
    print(f"Predykcje: {pred}")
    print("==================")

    return predicted_label, confidence, pred


if __name__ == "__main__":
    predict_and_show()
