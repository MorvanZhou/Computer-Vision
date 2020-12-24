import numpy as np
import os
from urllib.request import urlretrieve
from tensorflow import keras

MNIST_PATH = "./mnist.npz"


def load_mnist(path="./mnist.npz", norm=True):
    if not os.path.isfile(path):
        print("not mnist data is found, try downloading...")
        urlretrieve("https://s3.amazonaws.com/img-datasets/mnist.npz", path)
    with np.load(path, allow_pickle=True) as f:
        x_train = f['x_train'].astype(np.float32)[:, :, :, None]
        y_train = f['y_train'].astype(np.float32)[:, None]
        x_test = f['x_test'].astype(np.float32)[:, :, :, None]
        y_test = f['y_test'].astype(np.float32)[:, None]
        if norm:
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)


def save_model_structure(model: keras.Model, path=None):
    if path is None:
        path = "visual/{}/{}_structure.png".format(model.name, model.name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        keras.utils.plot_model(model, show_shapes=True, expand_nested=True, dpi=150, to_file=path)
    except Exception as e:
        print(e)


def save_model_weights(model: keras.Model, path=None):
    if path is None:
        path = "visual/{}/model/net".format(model.name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_weights(path)


def load_model_weights(model: keras.Model, path=None):
    if path is None:
        path = "visual/{}/model/net".format(model.name)
    model.load_weights(path)