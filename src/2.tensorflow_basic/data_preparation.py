import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def preparing_dataset():

    # function to make random dataset
    def load_random_dataset():
        x = np.random.sample((100, 3))
        dataset = tf.data.Dataset.from_tensor_slices(x)
        return dataset

    # function to load dataset using tensorflow-datasets
    def load_dataset_from_tensorflow():
        ds = tfds.load("mnist", split="train", shuffle_files=True)
        return ds

    # function to load dataset using keras module
    def load_dataset_from_keras_module():
        data_train, data_test = tf.keras.datasets.mnist.load_data()
        (images_train, labels_train) = data_train
        (images_test, labels_test) = data_test

        return images_train, labels_train, images_test, labels_test

    # function to load dataset using url
    def load_dataset_from_url():
        url = "https://storage.googleapis.com/download.tensorflow.org/data/illiad/butler.txt"
        text_path = tf.keras.utils.get_file("butler.txt", origin=url)
        print(text_path)


    def main():
        #dataset = load_random_dataset()
        #ds = load_dataset_from_tensorflow()
        #images_train, labels_train, images_test, labels_test = load_dataset_from_keras_module()
        load_dataset_from_url()



    main()

preparing_dataset()

'''# dataset from tensorflow-datasets
ds = tfds.load("mnist", split="train", shuffle_files=True)
print(ds)

# dataset from keras module
data_train, data_test = tf.keras.datasets.mnist.load_data()
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test

# dataset from online
url = "https://storage.googleapis.com/download.tensorflow.org/data/illiad/butler.txt"
text_path = tf.keras.utils.get_file("butler.txt", origin=url)'''