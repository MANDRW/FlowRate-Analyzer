import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.utils import load_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

def samples(images, labels, quantity=9):
    plt.figure(figsize=(12, 16))
    for i in range(quantity):
        idx = rand.randint(0, len(images) - 1)
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Klasa: {labels[idx]}")
        plt.axis("off")
    plt.show()

class DataPrep:
    def __init__(self,path,img_size=(512,512),batch_size=32,test_size=0.2,random_state=42,crop_ratio=0.5):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.crop_ratio=crop_ratio
        self.label_map = {"100":0,"010":1,"001":2} # 100 - under, 010 - good, 001 - over

    def crop(self,image,crop_ratio=0.5):
        h, w, _ = image.shape
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        left = max(0, (w - new_w) // 2)
        top = max(0, (h - new_h) // 2)
        right = left + new_w
        bottom = top + new_h
        return image[top:bottom, left:right]

    def load(self, crop=False):
        images = []
        labels = []
        for img in os.listdir(self.path):
            try:
                if (img.endswith(".png") or img.endswith(".PNG")):
                    path = os.path.join(self.path, img)
                    print(img)
                    file_label = img.split("_")[1].split(".")[0]
                    if(crop==True):
                        image=load_img(path)
                        image=np.array(image)
                        image=self.crop(image,self.crop_ratio)
                        image=Image.fromarray(image).resize(self.img_size)
                    else:
                        image = load_img(path, target_size=self.img_size)
                    image=np.array(image)/255.0
                    label = self.label_map[file_label]
                    images.append(image)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
                continue
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def prep(self,augment=True,crop=False):
        images, labels = self.load(crop)
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels)
        y_train = to_categorical(y_train, num_classes=3)
        y_val = to_categorical(y_val, num_classes=3)
        if augment:
            train_datagen = ImageDataGenerator(
                rotation_range=5,#5
                width_shift_range=0.2, #0.15
                height_shift_range=0.2,
                shear_range=5,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect',
                rescale=1. / 255.0,
                zoom_range=0.2
            )
        else:
            train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow(x_train, y_train, batch_size=self.batch_size)
        val_gen= val_datagen.flow(x_val, y_val, batch_size=self.batch_size)
        return train_gen, val_gen


if __name__ == "__main__":

    path = "close_flowrate_dataset"
    img_size = (4032//6, 3024//6)
    batch_size = 16
    test_size = 0.2
    random_state = 42
    crop_ratio=0.99 #0.6


    data_prep = DataPrep(path, img_size, batch_size, test_size, random_state,crop_ratio)

    train_gen, val_gen = data_prep.prep(augment=True,crop=True)

    x_batch, y_batch = next(train_gen)

    samples(x_batch, y_batch, quantity=9)