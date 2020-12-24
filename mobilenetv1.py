# [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# (dw+pw). small but effective cnn optimized on mobile (cpu)
def block(x, filters):
    # Depthwise Separable convolutions
    o = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(x)     # [n, h, w, c] dw
    o = layers.ReLU()(o)
    o = layers.Conv2D(filters, 1, 1)(o)                                         # [n, h, w, f] pw
    o = layers.ReLU()(o)
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=20)                                        # [n, 14, 14, 20]
    x = layers.MaxPool2D(2, 2)(x)                                   # [n, 7, 7, 20]
    x = block(x, 40)                                                # [n, 7, 7, 40]
    x = layers.GlobalAveragePooling2D()(x)                          # [n, 40]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="MobileNetV1")


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
