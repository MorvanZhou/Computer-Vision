# [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# Grouped 1x1 conv to reduce parameters, but shuffle grouped feature maps to mix isolated feature map.
def shuffle(x, groups):
    _, h, w, c = x.shape
    gc = c // groups
    o = layers.Reshape((h, w, groups, gc))(x)  # [n, h, w, g, gc]
    o = layers.Permute((1, 2, 4, 3))(o)         # shuffle in groups
    o = layers.Reshape((h, w, c))(o)
    return o


def group_conv(x, filters, groups):
    assert x.shape[-1] % groups == 0
    assert filters % groups == 0
    if tf.test.is_built_with_gpu_support():
        o = layers.Conv2D(filters, 1, 1, groups=groups)(x)  # [n, h, w, f] pw, groups=groups not works on cpu (tf=2.3.0)
    else:
        o = layers.concatenate([
            layers.Conv2D(filters // groups, 1, 1)(g) for g in tf.split(x, groups, axis=-1)
        ], axis=-1)  # this works on cpu        [n, h, w, f]
    return o


def block(x, filters, groups=4):
    o = group_conv(x, x.shape[-1], groups)          # [n, h, w, c] gpw
    o = shuffle(o, groups)
    o = layers.ReLU()(o)
    o = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(o)  # [n, h, w, c] dw
    o = group_conv(o, filters, groups)              # [n, h, w, f] gpw
    if x.shape[-1] != filters:
        x = group_conv(x, filters, groups)      # [n, h, w, f]
    o = layers.add((o, x))  # residual connection
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=32, groups=4)                              # [n, 14, 14, 32]
    x = layers.MaxPool2D(2, 2)(x)                                   # [n, 7, 7, 32]
    x = block(x, 64, 4)                                             # [n, 7, 7, 64]
    x = layers.GlobalAveragePooling2D()(x)                          # [n, 64]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="ShuffleNetV1")


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
