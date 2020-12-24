# [Gradient-Based Learning Applied to Document Recognition](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf)
# dependency file: https://github.com/MorvanZhou/Computer-Vision/requirements.txt

from tensorflow import keras
from tensorflow.keras import layers
from utils import load_mnist, save_model_structure, save_model_weights

# get data
(x_train, y_train), (x_test, y_test) = load_mnist()

# define model
model = keras.Sequential([
    layers.Conv2D(filters=8, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1)),   # [n, 28, 28, 8]
    layers.MaxPool2D(pool_size=2, strides=2),                             # [n, 14, 14, 8]
    layers.Conv2D(16, 3, 1, "same"),                                      # [n, 14, 14, 16]
    layers.MaxPool2D(2, 2),                                               # [n, 7, 7, 16]
    layers.Flatten(),                                                     # [n, 7*7*16]
    layers.Dense(10)                                                      # [n, 10]
], name="LeNet")

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


