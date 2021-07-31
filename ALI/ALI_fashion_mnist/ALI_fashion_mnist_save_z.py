import keras.backend as K
from keras.datasets import fashion_mnist
import numpy as np
import models
import pickle


def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    return x


def decode_output(x):
    x = x.astype(np.float32)
    x *= 255.
    return x


def reconstruct_img(x, labels, xgen, zgen):
    """
    x assumes x_train
    xgen: trained xgenerater
    zgen: trained zgenerater
    """
    # original images
    #ind = np.random.permutation(len(x))
    num_generate_imgs = 64
    #x = (x[ind])[:num_generate_imgs//2]
    x = x.astype(np.uint8)
    #labels=labels[ind]
    #labels=labels[:num_generate_imgs//2]
    # generated images
    x_copy = np.copy(x)
    x_copy = x_copy.astype(np.float32)
    x_copy = preprocess_input(x_copy)
    z_gen = zgen.predict_on_batch([x_copy, labels])
    x_gen = xgen.predict_on_batch([z_gen, labels])
    x_gen = decode_output(x_gen)
    x_gen = np.clip(x_gen, 0., 255.).astype(np.uint8)

    #grid_size = int(np.sqrt(num_generate_imgs))
    #cols = []
    #for i in range(0, num_generate_imgs//2, grid_size):
    #    col_orig = np.concatenate(x[i:i+grid_size], axis=0)
    #    col_gen = np.concatenate(x_gen[i:i+grid_size], axis=0)
    #    col = np.concatenate([col_orig, col_gen], axis=1)
    #    cols.append(col)
    #concatenated = np.concatenate(cols, axis=1)
    return x_gen


n_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, 3)
x_train = preprocess_input(x_train)
zgen = models.create_zgenerater(input_shape, n_classes)
zgen.load_weights('result/epochs300/zgen_weights.h5')
xgen = models.create_xgenerater(n_classes)
xgen.load_weights('result/epochs300/xgen_weights.h5')
batch_size = 64
z_train = []
i = 0
while (i + batch_size) < x_train.shape[0]:
    start_index = i
    end_index = start_index + batch_size
    z_train.append(
        reconstruct_img(x_train[start_index:end_index, :, :, :],
                        y_train[start_index:end_index], xgen, zgen))
    i = end_index
z_train.append(
    reconstruct_img(x_train[i:x_train.shape[0], :, :, :],
                    y_train[i:x_train.shape[0]], xgen, zgen))
z_train = np.vstack(z_train)
with open('recons_train_fashion.pkl', 'wb') as f:
    pickle.dump(z_train, f)
with open('recons_train_labels_fashion.pkl', 'wb') as f:
    pickle.dump(y_train, f)
