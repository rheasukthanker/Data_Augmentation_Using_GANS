from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pickle
from keras.models import load_model
from numpy import asarray
from numpy.random import randn
import numpy as np
import tensorflow as tf


def get_images_from_generator(label, model_name, num_samples, latent_dim):
    model = load_model(model_name)
    latent_points, labels = generate_latent_points(latent_dim, num_samples,
                                                   label)
    X = model.predict([latent_points, labels])
    return [X, labels]


def generate_latent_points(latent_dim, num_samples, label):
    # generate points in the latent space
    x_input = randn(latent_dim * num_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(num_samples, latent_dim)
    # generate labels
    labels = np.repeat(label, num_samples)
    return [z_input, labels]


batch_size = 64
epochs = 20
IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_CHANNELS = 3
NUM_CLASSES = 2
latent_dim = 100
model = Sequential([
    Conv2D(16,
           3,
           padding='same',
           activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=4)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_med.hdf5',
                                              save_best_only=True,
                                              monitor='val_loss',
                                              mode='min')
with open("../../data/all_images.pkl", "rb") as f:
    all_images = pickle.load(f)
with open("../../data/all_labels.pkl", "rb") as f:
    all_labels = pickle.load(f)
randomize = np.arange(np.shape(all_images)[0])
np.random.shuffle(randomize)
X = all_images[randomize, :, :, :]
y = all_labels[randomize]
del all_images
del all_labels
train_images, test_images, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=42)
num_positives = np.sum(y == 1)
num_negatives = np.sum(y == 0)
train_images = (train_images - 127.5) / 127.5
test_images = (test_images - 127.5) / 127.5
gan_samples = []
gan_labels = []
train_labels = np.reshape(train_labels, [np.shape(train_labels)[0], 1])
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)
i = 1
[X,
 labels] = get_images_from_generator(i,
                                     "../../data/CGAN/cgan_generator_med.h5",
                                     num_negatives - num_positives, latent_dim)
gan_samples.append(X)
gan_labels.append(np.expand_dims(labels, 1))
gan_samples.append(train_images)
gan_labels.append(train_labels)
gan_samples = np.vstack(gan_samples)
gan_labels = np.vstack(gan_labels)
inds = np.random.permutation(np.shape(gan_samples)[0])
gan_samples = gan_samples[inds, :, :, :]
gan_labels = gan_labels[inds]
model.fit(gan_samples,
          gan_labels,
          batch_size=batch_size,
          callbacks=[early_stopping, mcp_save],
          validation_data=(val_images, val_labels),
          epochs=epochs)
model.load_weights("best.mdl_wts_med.hdf5")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
