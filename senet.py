# [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# SE is a module that learns to scale each feature map, we can plug in a resnet block
# larger reduction_ratio reduce parameter size in FC layers with limited accuracy drop
def se(x, reduction_ratio=4):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(c // reduction_ratio)(s)
    s = layers.ReLU()(s)
    s = layers.Dense(c, activation=keras.activations.sigmoid)(s)
    return x * layers.Reshape((1, 1, c))(s)


def bottleneck(x, filters, strides=1, reduction_ratio=4):
    in_channels = x.shape[-1]
    if in_channels != filters:
        shortcut = layers.Conv2D(filters, 1, strides, "same", name="projection")(x)
    else:
        shortcut = x
    o = layers.ReLU()(x)
    o = layers.Conv2D(in_channels, kernel_size=3, strides=1, padding="same")(o)     # [n, h, w, c]
    o = layers.ReLU()(o)
    o = layers.Conv2D(filters, 3, strides, "same")(o)                               # [n, h/s, w/s, f]
    o = se(o, reduction_ratio)
    o = layers.add((o, shortcut))
    return o


def block(x, filters, strides=1, n_bottleneck=2):
    o = bottleneck(x, filters, strides)     # [n, h/s, w/s, f]
    for i in range(1, n_bottleneck):
        o = bottleneck(o, filters, 1)       # [n, h/s, w/s, f]
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=8, strides=1, n_bottleneck=2)              # [n, 14, 14, 8]
    x = block(x, 16, 2, 1)                                          # [n, 7, 7, 16]
    x = layers.GlobalAveragePooling2D()(x)                            # [n, 16]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="SENet")

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
