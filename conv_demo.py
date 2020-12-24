import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils import load_mnist


def save_gif(imgs_dir):
    imgs = []
    imgs_path = [os.path.join(imgs_dir, p) for p in os.listdir(imgs_dir) if p.endswith(".png")]
    for f in sorted(imgs_path, key=os.path.getmtime):
        if not f.endswith(".png"):
            continue
        img = Image.open(f)
        img = img.resize((img.width, img.height), Image.ANTIALIAS)
        imgs.append(img)
    path = "{}/conv_mechanism.gif".format(os.path.dirname(imgs_dir))
    if os.path.exists(path):
        os.remove(path)
    imgs[0].save(path, append_images=imgs[1:], optimize=False, save_all=True, duration=20, loop=0)
    print("saved ", path)


def show_conv(save_dir, image):
    filter = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]])
    plt.figure(0, figsize=(9, 5))
    ax1 = plt.subplot(121)
    ax1.imshow(image, cmap='gray_r')
    plt.xticks(())
    plt.yticks(())
    ax2 = plt.subplot(122)
    texts = []
    feature_map = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):

            if texts:
                fm.remove()
            for n in range(3):
                for m in range(3):
                    if len(texts) != 9:
                        texts.append(ax1.text(j+m, i+n, filter[n, m], color='k', size=8, ha='center', va='center'))
                    else:
                        texts[n*3+m].set_position((j+m, i+n))

            feature_map[i, j] = np.sum(filter * image[i:i+3, j:j+3])
            fm = ax2.imshow(feature_map, cmap='gray', vmax=3, vmin=-3)
            plt.xticks(())
            plt.yticks(())
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "i{}j{}.png".format(i, j)))


def show_feature_map(save_dir, image):
    filters = [
        np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]]),
        np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]]),
        np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]]),
        np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]])
    ]

    plt.figure(1)
    plt.title('Original image')
    plt.imshow(image, cmap='gray_r')
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(save_dir, "original_img.png"))

    plt.figure(2)
    for n in range(4):
        feature_map = np.zeros((26, 26))

        for i in range(26):
            for j in range(26):
                feature_map[i, j] = np.sum(image[i:i + 3, j:j + 3] * filters[n])

        plt.subplot(3, 4, 1 + n)
        plt.title('Filter%i' % n)
        plt.imshow(filters[n], cmap='gray', vmax=3, vmin=-3)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(3, 4, 5 + n)
        plt.title('Conv%i' % n)
        plt.imshow(feature_map, cmap='gray', vmax=3, vmin=-3)
        plt.xticks(())
        plt.yticks(())

        plt.subplot(3, 4, 9 + n)
        plt.title('ReLU%i' % n)
        feature_map = np.maximum(0, feature_map)
        plt.imshow(feature_map, cmap='gray', vmax=3, vmin=-3)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_map.png"))


if __name__ == "__main__":
    result_dir = "visual/basic"
    conv_dir = os.path.join(result_dir, "conv")
    os.makedirs(conv_dir, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    data = x_train[7].squeeze(axis=-1)
    show_feature_map(result_dir, data)
    show_conv(conv_dir, data)
    save_gif(conv_dir)
