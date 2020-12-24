# [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# compared with resnet, it has less filter each conv, sees more previous inputs
def bottleneck(xs, growth_rate):
    if len(xs) == 1:
        o = xs[0]
    else:
        o = layers.concatenate(xs, axis=-1)         # [n, h, w, c * xs]
    o = layers.ReLU()(o)
    o = layers.Conv2D(growth_rate, kernel_size=1, strides=1)(o)  # [n, h, w, c]
    o = layers.ReLU()(o)
    o = layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding="same")(o)  # [n, h, w, c]
    return o


def block(x, growth_rate, n_bottleneck=2):
    outs = [bottleneck([x], growth_rate)]   # [n, h, w, c]
    for i in range(1, n_bottleneck):
        o = bottleneck(outs, growth_rate)        # [n, h, w, c]
        outs.append(o)
    return layers.concatenate(outs, axis=-1)    # [n, h, w, c * n_bottleneck]


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = block(x, growth_rate=8, n_bottleneck=2)     # [n, 14, 14, 8*2]
    x = layers.MaxPool2D(2)(x)                      # [n, 7, 7, 8*2]
    x = block(x, 8, 3)                              # [n, 7, 7, 8*3]
    x = layers.GlobalAveragePooling2D()(x)          # [n, 24]
    o = layers.Dense(10)(x)                         # [n, 10]
    return keras.Model(inputs, o, name="DenseNet")

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
