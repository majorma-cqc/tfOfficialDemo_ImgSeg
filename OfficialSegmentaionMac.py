### This demo used UNET.


from os import wait
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras import callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.ops.gen_array_ops import concat
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



def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    '''if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    '''

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask



# The dataset already contains the required training and test splits, continue to use the same splits.
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE



train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image)



''' class Augment performs a simple augmentation(增长 扩大) by randomly-flipping an image. '''
### This tutorial demonstrates data augmentation: a technique to increase the diversity 
# of your training set by applying random (but realistic) transformations, such as image rotation.
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they will make the same random changes

        ### Preprocessing:
        #   Keras 预处理层(preprocessing) API 允许开发人员构建 Keras 原生输入处理管道。这些输入处理管道可用作非 Keras 工作流程中的独立预处理代码，
        # 直接与 Keras 模型结合，并作为 Keras SavedModel 的一部分导出。
        self.augment_inputs = preprocessing.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = preprocessing.RandomFlip(mode="horizontal", seed=seed)
    
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels



''' Build the input pipeline, applying the Augmentation after batching the inputs. '''
train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)



''' Visualize an image example and its corresponding mask from the dataset. '''
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()



for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])



''' Define the model '''
#_________________________________________________________________________________________________________#
### The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). 
# In-order to learn robust features and reduce the number of trainable parameters, you will use a pretrained model - MobileNetV2 - as the encoder. 
# For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples repo.

### As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in tf.keras.applications. 
# The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.
#_________________________________________________________________________________________________________#

#  Encoder, pretrained.
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False)
#  Use the activations of these layers:
layer_names = [
    'block_1_expand_relu',   # 64 64
    'block_3_expand_relu',   # 32 32
    'block_6_expand_relu',   # 16 16
    'block_13_expand_relu',  # 8 8
    'block_16_project',      # 4 4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#  Create feature extraction model.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False



#  Decoder/upsampler just sefies of upsampling blocks.
up_stack = [
    pix2pix.upsample(512, 3), # 4 4 -> 8 8
    pix2pix.upsample(256, 3), # 8 8 -> 16 16
    pix2pix.upsample(128, 3), # 16 16 -> 32 32
    pix2pix.upsample(64, 3),  # 32 32 -> 64 64
]



def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    #  Downsampling
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #  Upsampling && establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    
    #  This is the last layer.
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same'
    )  # 64 64 -> 128 128

    return tf.keras.Model(inputs=inputs, outputs=x)

### Note that the number of filters on the last layer is set to the number of output_channels. 
# This will be one output channel per class.



''' Train the model '''
### Compile + train
# tf.keras.losses.CategoricalCrossentropy loss function with the from_logits argument set to True.
# because labelss are scalar integers.
OUTPUT_CLASSES = 3
mdl = unet_model(output_channels=OUTPUT_CLASSES)
mdl.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])



''' Resulting model architechture: '''
tf.keras.utils.plot_model(mdl, show_shapes=True)



''' Try out mdl to check : before training '''
def create_mase(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]



def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = mdl.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
        else:
            display([sample_image, sample_mask,
            create_mask(mdl.predict(sample_image[tf.newaxis, ...]))])



show_predictions()



### The callback defined below is used to observe how the model improves while it is training.
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after each epoch {}\n'.format(epoch+1))



EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

model_history = mdl.fit(train_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS, validation_data=test_batches,
                        callbacks=[DisplayCallback()])



loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()



''' Make predictions '''
show_predictions(test_batches, 3)