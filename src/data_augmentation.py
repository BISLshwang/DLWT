from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img = load_img("/usr/local/anaconda3/envs/DLWT/src/5.cnn/data/bird.jpg")
data = img_to_array(img)

# width_shift_range
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(width_shift_range=[-200, 200])
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()

# height_shift_range
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(height_shift_range=0.5)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()

# flip
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()

# rotation
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(rotation_range=90)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()

# brightness
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(brightness_range=[0.3, 1.2])
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()

# zoom
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(zoom_range=0.1)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)

plt.show()
