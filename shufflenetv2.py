# [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# Further reduces parameters by switching group conv with split+concat, perform shuffle at end of block.
# Speed up calculation.
def shuffle(x, groups):
    _, h, w, c = x.shape
    gc = c // groups
    o = layers.Reshape((h, w, groups, gc))(x)  # [n, h, w, g, gc]
    o = layers.Permute((1, 2, 4, 3))(o)         # shuffle in groups
    o = layers.Reshape((h, w, c))(o)
    return o


def block(x, filters):
    x1, x2 = tf.split(x, 2, axis=-1)            # x1 [n, h, w, c/2], x2 [n, h, w, c/2]
    c = x.shape[-1]
    o = layers.Conv2D(x2.shape[-1], 1, 1)(x2)   # [n, h, w, c/2]
    o = layers.ReLU()(o)
    o = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(o)     # [n, h, w, c/2]
    o = layers.ReLU()(o)
    o = layers.Conv2D(filters//2, 1, 1)(o)          # [n, h, w, f/2]
    if filters != c:
        x1 = layers.Conv2D(filters//2, 1, 1)(x1)    # [n, h, w, f/2]
    o = layers.concatenate([x1, o], axis=-1)        # [n, h, w, f]
    o = shuffle(o, 2)   #
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=32)                                        # [n, 14, 14, 32]
    x = layers.MaxPool2D(2, 2)(x)                                   # [n, 7, 7, 32]
    x = block(x, 64)                                                # [n, 7, 7, 64]
    x = layers.GlobalAveragePooling2D()(x)                          # [n, 64]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="ShuffleNetV2")


# show model
model = build_model()
model.summary()
save_model_structure(model)

# define loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = keras.optimizers.Adam(0.001)
accuracy = keras.metrics.SparseCategoricalAccuracy()
model.compile(optimizer=opt, loss=loss, metrics=[accuracy])

# training and validation
model.fit(x=x_train, y=y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))

# save model
save_model_weights(model)
