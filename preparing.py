import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

def samples(images,labels,quantity=9):
    plt.figure(figsize=(10, 10))
    for i in range(quantity):
        plt.subplot(3, 3, i + 1)
        num=rand.randint(0, len(images) - 1)
        img = load_img(images[num])
        plt.imshow(img)
        plt.title(num)
        plt.axis("off")
    plt.show()


class DataPrep:
    def __init__(self,path,img_size=(512,512),batch_size=32,test_size=0.2,random_state=42):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.label_map = {"100":0,"010":1,"001":2} # 100 - under, 010 - good, 001 - over

    def load(self):
        images = []
        labels = []
        for img in os.listdir(self.path):
            if (img.endswith(".png") or img.endswith(".PNG")):
                path = os.path.join(self.path, img)
                img=load_img(path,target_size=self.img_size)
                img=np.array(img)/255.0
                label = [int(x) for x in img.split("_")[1].split(".")[0]]
                label=self.label_map(label)
                images.append(img)
                labels.append(label)
        images = np.array(images)
        labels = to_categorical(labels,num_classes=3)
        return images, labels

    def prep(self,augment=True):
        images, labels = self.load()
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels)
        if(augment==True):
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        val_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow(x_train, y_train, batch_size=self.batch_size)
        val_gen= val_datagen.flow(x_val, y_val, batch_size=self.batch_size)
        return train_gen, val_gen
