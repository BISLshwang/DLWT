# 1. import library

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# check gpu connection
#tf.debugging.set_log_device_placement(True)
#tf.config.list_physical_devices('GPU')

# check logging device placement
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


# define model using SequentialAPI
# CNN
# data preprocessing

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(100, 100, 3), activation="relu", kernel_size=(5,5), filters=32),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation="relu", kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation="relu", kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation="relu", kernel_size=(5, 5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.summary()

# define feature map
# input: (None, 100, 100, 3)
ins = model.inputs
# ouput of first layer: (None, 96, 96, 32)
outs = model.layers[0].output
feature_map = Model(inputs=ins, outputs=outs)
feature_map.summary()

# load image
img = cv2.imread("/opt/project/data/cat.jpg")
plt.imshow(img)

# image preprocessing and feature map generation
img = cv2.resize(img, (100, 100))
input_img = np.expand_dims(img, axis=0)
print(input_img.shape)

feature = feature_map.predict(input_img)
print(feature.shape)

fig = plt.figure(figsize=(50, 50))
for i in range(16):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(feature[0,:,:,i])

# ouput of second layer: (None, 1000)
outs = model.layers[2].output
feature_map = Model(inputs=ins, outputs=outs)
feature_map.summary()

feature = feature_map.predict(input_img)
print(feature.shape)

fig = plt.figure(figsize=(50, 50))
for i in range(48):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(feature[0,:,:,i])

# ouput of second layer: (None, 1000)
outs = model.layers[6].output
feature_map = Model(inputs=ins, outputs=outs)
feature_map.summary()

feature = feature_map.predict(input_img)
print(feature.shape)

fig = plt.figure(figsize=(50, 50))
for i in range(48):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(feature[0,:,:,i])