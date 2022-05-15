import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import model_builder
import yaml_config

config = yaml_config.takeConfig("../config/train_model.yaml")

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
img_path = config["path"]["img"]
mask_path = config["path"]["mask"]
validation_img_path = config["path"]["val_img"]
validation_mask_path = config["path"]["val_mask"]

NO_OF_EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
NO_OF_TRAINING_IMAGES = len(os.listdir(img_path))
NO_OF_VAL_IMAGES = len(os.listdir(validation_img_path))

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder)  # List of training images
    random.shuffle(n)

    while True:
        img = np.zeros((batch_size, 512, 512, 1)).astype("float")
        mask = np.zeros((batch_size, 512, 512, 1)).astype("float")

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.

            img1 = plt.imread(img_folder + "/" + n[i])
            mask1 = plt.imread(mask_folder + "/" + n[i])

            img1 = cv2.resize(img1, (512, 512))
            mask1 = cv2.resize(mask1, (512, 512))

            img1 = np.asarray(img1)
            mask1 = np.asarray(mask1)

            img1 = np.expand_dims(img1, axis=2)
            mask1 = np.expand_dims(mask1, axis=2)

            img[i - c] = img1
            mask[i - c] = mask1

        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            random.shuffle(n)

        yield img, mask


train_gen = data_gen(img_path, mask_path, BATCH_SIZE)
val_gen = data_gen(validation_img_path, validation_mask_path, BATCH_SIZE)

# Build the model
model = model_builder.unet()


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(
        y_pred, axis=[1, 2, 3]
    )
    return tf.keras.backend.mean(
        (2.0 * intersection + smooth) / (union + smooth), axis=0
    )


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


model.compile(optimizer="adam", loss=dice_loss, metrics=["accuracy", dice_coef])

retVal = model.fit(
    train_gen,
    epochs=NO_OF_EPOCHS,
    steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
    validation_data=val_gen,
    validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE),
)
