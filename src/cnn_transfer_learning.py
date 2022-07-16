#1. import library
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import tensorflow_hub as hub

# check gpu connection
#tf.debugging.set_log_device_placement(True)
#tf.config.list_physical_devices('GPU')

# check logging device placement
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

# 2. load pre-trained model
model = ResNet50(include_top=True,
                 weights="imagenet",
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000)

# 3. check architecture
model.summary()

# 4. add fully coneected layer
model.trainable = False
model = Sequential([model,
                    Dense(2, activation='sigmoid')])
model.summary()

# using tensorflow_hub
'''model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4",
                   input_shape=(224, 224, 3),
                   trainable=False),
    tf.keras.layers.Dense(2, activation='softmax')
])'''


# 5. model compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 6. model training
BATCH_SIZE = 32
image_height = 224
image_width = 224
# classify cat and dog from kaggle
# dog=12501, cat=12501
train_dir = "/opt/project/data/catanddog/train"
valid_dir = "/opt/project/data/catanddog/valildation"

# data preprocessing uisng ImageDataGenerator
train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

train_generator = train.flow_from_directory(train_dir,
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            shuffle=True,
                                            class_mode="categorical")

valid = ImageDataGenerator(rescale=1./255.)

valid_generator = valid.flow_from_directory(valid_dir,
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=7,
                                            shuffle=True,
                                            class_mode="categorical")

history = model.fit(train_generator,
                    epochs=10,
                    validation_data=valid_generator,
                    verbose=1)

# 7. model performance visualization
accuracy = history.history["accuracy"]
val_accuracy = history.history['val_accuracy']

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, label='training accuracy')
plt.plot(epochs, val_accuracy, lable='validation accuracy')
plt.legend()
plt.title("accuracy")

plt.figure()
plt.plot(epochs, loss, label="training loss")
plt.plot(epochs, val_loss, label="validation accuracy")
plt.legend()
plt.title("loss")

# 8. prediction of the model

class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))
prediction_values = model.predict(validation)
prediction_values = np.argmax(prediction_values, axis=1)

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color="yellow", fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color="red", fontsize=14)
