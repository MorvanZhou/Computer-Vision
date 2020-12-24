# [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# (residual+pw+dw+pw). mobilenet v1 with residual block and layer rearrange
def block(x, filters, expand_ratio=4):
    # mobilenet v1: dw > pw
    # mobilenet v2: pw > dw > pw    let dw see more feature maps
    o = layers.Conv2D(int(filters*expand_ratio), 1, 1)(x)                       # [n, h, w, c*e]  pw expansion
    o = layers.ReLU()(o)
    o = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(o)     # [n, h/s, w/s, c*e] dw
    o = layers.ReLU()(o)
    o = layers.Conv2D(filters, 1, 1)(o)                         # [n, h, w, c] pw
    if x.shape[-1] == filters:
        o = layers.add((o, x))                                  # residual connection
    return o


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, filters=5)                                            # [n, 14, 14, 5]
    x = layers.MaxPool2D(2, 2)(x)                                   # [n, 7, 7, 5]
    x = block(x, 10)                                                   # [n, 7, 7, 10]
    x = layers.GlobalAveragePooling2D()(x)                          # [n, 10]
    o = layers.Dense(10)(x)                                         # [n, 10]
    return keras.Model(inputs, o, name="MobileNetV2")


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
