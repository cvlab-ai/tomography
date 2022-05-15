import os
from typing import Dict, Union, List, Any

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import glob
import model_builder
import yaml_config
import uuid

config = yaml_config.takeConfig("../config/cnn.yaml")

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
img_path = config["path"]["img"]
mask_path = config["path"]["mask"]
validation_img_path = config["path"]["val_img"]
validation_mask_path = config["path"]["val_mask"]
prefix_path = config["path"]["checkpoint"]

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

framObjTrain: Dict[str, Union[Union[List[Any], List[str]], Any]] = {
    "img": [],
    "mask": [],
}

framObjValidation: Dict[str, Union[Union[List[Any], List[str]], Any]] = {
    "img": [],
    "mask": [],
}


def LoadData(frameObj=None, imgPath=None, maskPath=None, shape=256):
    imgNames = os.listdir(imgPath)

    imgAddr = imgPath + "/"
    maskAddr = maskPath + "/"

    for i in range(len(imgNames)):  # range(100):

        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + imgNames[i])

        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))

        img = np.asarray(img)
        mask = np.asarray(mask)

        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)

        frameObj["img"].append(img)
        frameObj["mask"].append(mask)
        # print(i)

    return frameObj


print("before load")

framObjTrain = LoadData(framObjTrain, imgPath=img_path, maskPath=mask_path, shape=512)

framObjValidation = LoadData(
    framObjValidation,
    imgPath=validation_img_path,
    maskPath=validation_mask_path,
    shape=512,
)

print("after load")

plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.imshow(framObjTrain["img"][55])
plt.title("Image")
plt.subplot(1, 2, 2)
plt.imshow(framObjTrain["mask"][55])
plt.title("Mask")
plt.show()

# Build the model


if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)

checkpoint_path = prefix_path + "{epoch:04d}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
list_of_files = glob.glob(prefix_path + "*.hdf5")
latest: str = max(list_of_files, key=os.path.getctime, default=0)  # type: ignore
exists = latest and os.path.isfile(latest)

if exists:
    model = model_builder.unet(latest)
    print("Weights from checkpoint file loaded")
else:
    model = model_builder.unet()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", verbose=1, mode="max", save_freq="epoch"
)

callbacks_list = [checkpoint]
model.summary()

initial_epoch = 0
if exists:
    initial_epoch = int(latest.replace(prefix_path, "").replace(".hdf5", ""))
    print("Start from epoch", initial_epoch)

retVal = model.fit(
    np.array(framObjTrain["img"]),
    np.array(framObjTrain["mask"]),
    epochs=2000,
    verbose=1,
    validation_split=0.1,
    batch_size=12,
    shuffle=True,
    callbacks=callbacks_list,
    initial_epoch=initial_epoch,
    workers=2,
    use_multiprocessing=True,
)
#  validation_data=(np.array(framObjValidation['img']), np.array(framObjValidation['mask']))
plt.figure()
plt.plot(retVal.history["accuracy"])
plt.plot(retVal.history["val_accuracy"])
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"], loc="upper left")
plt.savefig("accvsepochs.png")

plt.figure()
plt.plot(retVal.history["loss"])
plt.plot(retVal.history["val_loss"])
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Val"], loc="upper right")
plt.savefig("loss.png")

plt.figure()
plt.plot(retVal.history["loss"], label="training_loss")
plt.plot(retVal.history["accuracy"], label="training_accuracy")
plt.legend()
plt.grid(True)


def predict16(valMap, model, shape=512):
    # getting and proccessing val data
    img = valMap["img"][0:16]
    mask = valMap["mask"][0:16]

    imgProc = img[0:16]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    name = uuid.uuid4()
    plt.figure(figsize=(9, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(" image")

    plt.subplot(1, 3, 2)
    plt.imshow(predMask)
    plt.title("Predicted mask")

    plt.subplot(1, 3, 3)
    plt.imshow(groundTruth)
    plt.title("Actual mask")
    plt.savefig(str(name) + ".png")


sixteenPrediction, actuals, masks = predict16(framObjTrain, model)
Plotter(actuals[1], sixteenPrediction[1][:, :, 0], masks[1])

Plotter(actuals[2], sixteenPrediction[2][:, :, 0], masks[2])

Plotter(actuals[3], sixteenPrediction[3][:, :, 0], masks[3])

model.save("Segmentor.h5")
