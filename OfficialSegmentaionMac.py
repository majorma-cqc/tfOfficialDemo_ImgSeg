import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt


''' Download Oxford-IIIT Pets dataset '''
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


### Image color values are normalized to the [0,1] range.
# The pixels in teh seg mask are labeled {0, 1, 2}
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # For the image default mask = {1, 2, 3}, change into {0,1,2} is easy to operate.
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoin):
    return 0

# The dataset already contains the required training and test splits, continue to use the same splits.

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# 这什么
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)