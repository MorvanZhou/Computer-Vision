# [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()

# define model
# like LeNet with more layers and activations
model = keras.Sequential([
    layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1)),   # [n, 28, 28, 8]
    layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1)),   # [n, 28, 28, 8]
    layers.ReLU(),
    layers.MaxPool2D(pool_size=2, strides=2),                             # [n, 14, 14, 8]
    layers.Conv2D(16, 3, 1, "same"),                                      # [n, 14, 14, 16]
    layers.Conv2D(16, 3, 1, "same"),                                      # [n, 14, 14, 16]
    layers.ReLU(),
    layers.MaxPool2D(2, 2),                                               # [n, 7, 7, 16]
    layers.Flatten(),                                                     # [n, 7*7*16]
    layers.Dense(32),                                                     # [n, 32]
    layers.ReLU(),
    layers.Dense(10)                                                      # [n, 32]
], name="VGG")

# show model
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
