# [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# (residual+pw+dw) just like mobilenetv2 without last pw
def block(x, filters):
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, 1)(x)
    else:
        shortcut = x
    o = layers.ReLU()(x)
    o = layers.Conv2D(filters, 1, 1)(o)                 # [n, h, w, f]  pw
    o = layers.ReLU()(o)
    o = layers.DepthwiseConv2D(3, 1, "same")(o)         # [n, h, w, f]  dw
    o = layers.add((o, shortcut))  # residual connection
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=16)                                           # [n, 14, 14, 16]
    x = layers.MaxPool2D(2, 2)(x)                                   # [n, 7, 7, 16]
    x = block(x, 32)                                                   # [n, 7, 7, 32]
    x = layers.GlobalAveragePooling2D()(x)                          # [n, 32]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="Xception")


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
