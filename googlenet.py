# [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers, activations
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()


# multi kernel size to capture different local information
def inception(x, f1, f2, f3, f4, name):
    act = activations.relu
    inputs = layers.Input(shape=x.shape[1:])

    p1 = layers.Conv2D(filters=f1, kernel_size=1, strides=1, activation=act, name="p1")(inputs)

    p2 = layers.Conv2D(f2[0], 1, 1, name="p21")(inputs)
    p2 = layers.Conv2D(f2[1], 3, 1, padding="same", activation=act, name="p22")(p2)

    p3 = layers.Conv2D(f3[0], 1, 1, name="p31")(inputs)
    p3 = layers.Conv2D(f3[1], 5, 1, padding="same", activation=act, name="p32")(p3)

    p4 = layers.MaxPool2D(pool_size=3, strides=1, padding="same", name="p41")(inputs)
    p4 = layers.Conv2D(f4, 1, activation=act, name="p42")(p4)

    p = layers.concatenate((p1, p2, p3, p4), axis=-1)
    m = keras.Model(inputs, p, name=name)
    return m(x)


def build_model():
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(inputs)      # [n, 28, 28, 8]
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)                                     # [n, 14, 14, 8]
    x = inception(x, f1=4, f2=(2, 4), f3=(2, 4), f4=4, name="inspection1")              # [n, 14, 14, 4*4]
    x = layers.MaxPool2D(2, 2)(x)                                                       # [n, 7, 7, 4*4]
    x = inception(x, f1=8, f2=(4, 8), f3=(4, 8), f4=8, name="inspection2")              # [n, 7, 7, 8*4]
    x = layers.GlobalAveragePooling2D()(x)                                              # [n, 8*4]
    o = layers.Dense(10)(x)                                                             # [n, 10]
    return keras.Model(inputs, o, name="GoogLeNet")


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


