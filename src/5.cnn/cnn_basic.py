# 1. import library
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# check gpu connection
tf.debugging.set_log_device_placement(True)
tf.config.list_physical_devices('GPU')

# check logging device placement
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

# 2. load dataset using keras module
# fashion_mnist
# train=60000, test=10000
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 3. define class
# class num=10
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               'Sandal', 'Shirt', "Sneaker", "Bag", "Ankle boot"]

# show image examples
plt.figure(1)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

# 4. define and train model
x_train, x_test = x_train/255.0, x_test/255.0

# define model using SequentialAPI
# DNN
# for beginners, not for complicated models
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

# define model using SequentialAPI
# CNN
# data preprocessing

x_train_final = x_train.reshape((-1, 28, 28, 1))
x_test_final = x_test.reshape((-1, 28, 28, 1))
print(x_train_final.shape, x_test_final.shape)

model_with_conv = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(62, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_with_conv.compile(optimizer="adam",
                        loss='sparse_categorical_crossentropy',
                        metrics=["accuracy"])

model_with_conv.fit(x_train_final, y_train, epochs=5)
model_with_conv.evaluate(x_test_final, y_test, verbose=2)