import keras.backend as K
from keras.datasets import cifar100
import numpy as np
import models
import pickle
from sklearn.model_selection import train_test_split


def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    return x


def reconstruct_img(x, labels, zgen):
    """
    x assumes x_train
    xgen: trained xgenerater
    zgen: trained zgenerater
    """
    x_copy = np.copy(x)
    x_copy = x_copy.astype(np.float32)
    x_copy = preprocess_input(x_copy)
    z_gen = zgen.predict_on_batch([x_copy, labels])
    return z_gen


n_classes = 2
input_shape = (50, 50, 3)
with open("/cluster/scratch/srhea/all_images.pkl", "rb") as f:
    all_images = pickle.load(f)
with open("/cluster/scratch/srhea/all_labels.pkl", "rb") as f:
    all_labels = pickle.load(f)
randomize = np.arange(np.shape(all_images)[0])
np.random.shuffle(randomize)
X = all_images[randomize, :, :, :]
y = all_labels[randomize]
del all_images
del all_labels
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
x_train = preprocess_input(x_train)
zgen = models.create_zgenerater(input_shape, n_classes)
zgen.load_weights('result/epochs180/zgen_weights.h5')
batch_size = 64
z_train = []
i = 0
while (i + batch_size) < x_train.shape[0]:
    start_index = i
    end_index = start_index + batch_size
    z_train.append(
        reconstruct_img(x_train[start_index:end_index, :, :, :],
                        y_train[start_index:end_index], zgen))
    i = end_index
z_train.append(
    reconstruct_img(x_train[i:x_train.shape[0], :, :, :],
                    y_train[i:x_train.shape[0]], zgen))
z_train = np.vstack(z_train)
with open('z_train_med.pkl', 'wb') as f:
    pickle.dump(z_train, f)
with open('z_train_labels_med.pkl', 'wb') as f:
    pickle.dump(y_train, f)
