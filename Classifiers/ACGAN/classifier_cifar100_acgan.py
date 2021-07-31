from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from numpy.random import randn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


def get_images_from_generator(label, model_name, num_samples, latent_dim):
    model = load_model(model_name)
    latent_points, labels = generate_latent_points(latent_dim, num_samples,
                                                   label)
    X = model.predict([latent_points, labels])
    return [X, labels]


def generate_latent_points(latent_dim, num_samples, label):
    x_input = randn(latent_dim * num_samples)
    z_input = x_input.reshape(num_samples, latent_dim)
    labels = np.repeat(label, num_samples)
    return [z_input, labels]


def generate_imbalance(trainX, trainy, class_cur, perc):
    class_to_sparsify = class_cur
    inds_class = np.argwhere(trainy == class_to_sparsify)
    num_delete = int((1 - perc) * len(inds_class))
    index = np.random.RandomState(seed=42).choice(inds_class.shape[0],
                                                  num_delete,
                                                  replace=False)
    index_class = inds_class[index]
    trainX_sparsified = np.delete(trainX, index_class, axis=0)
    trainy_sparsified = np.delete(trainy, index_class)
    return trainX_sparsified, trainy_sparsified


#Obtaining arguments
parser = argparse.ArgumentParser(
    description='Classifier with GAN augmentation')
parser.add_argument(
    '--model_name',
    nargs="?",
    type=str,
    default='../../data/ACGAN_base/acgan_generator_cifar100.h5',
    help='model_name')
parser.add_argument('--num_epoch',
                    nargs="?",
                    type=int,
                    default=100,
                    help='number of epochs')
args = parser.parse_args()
model_name = args.model_name
epochs = args.num_epoch

batch_size = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 100
latent_dim = 64

#Classifier model
model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_cifar100.hdf5',
                                              save_best_only=True,
                                              monitor='val_loss',
                                              mode='min')

#Load real data and sparsify to match set on which GAN was trained
cifar100 = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
train_images = (train_images - 127.5) / 127.5
test_images = (test_images - 127.5) / 127.5
classes_to_sparsify = [
    83, 69, 35, 34, 2, 90, 37, 49, 23, 26, 41, 62, 33, 12, 84, 50, 57, 74, 93,
    40
]
sparsify_percentage = [
    0.65, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8
]
for i in range(len(classes_to_sparsify)):
    train_images, train_labels = generate_imbalance(train_images, train_labels,
                                                    classes_to_sparsify[i],
                                                    sparsify_percentage[i])
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1)

#Generate fake images
gan_samples = []
gan_labels = []
for i in range(len(classes_to_sparsify)):
    [X, labels
     ] = get_images_from_generator(classes_to_sparsify[i], str(model_name),
                                   int((1 - sparsify_percentage[i]) * 500),
                                   latent_dim)
    gan_samples.append(X)
    gan_labels.append(np.expand_dims(labels, 1))

#Augment data
gan_samples.append(train_images)
gan_labels.append(np.reshape(train_labels, [np.shape(train_labels)[0], 1]))
gan_samples = np.vstack(gan_samples)
gan_labels = np.vstack(gan_labels)
inds = np.random.permutation(gan_samples.shape[0])
gan_samples = gan_samples[inds, :, :, :]
gan_labels = gan_labels[inds]
gan_labels = np.squeeze(gan_labels)
gan_labels = np_utils.to_categorical(gan_labels, num_classes=100)
test_labels = np_utils.to_categorical(test_labels, num_classes=100)
val_labels = np_utils.to_categorical(val_labels, num_classes=100)

#Fit model of augmented dataset
model.fit(gan_samples,
          gan_labels,
          batch_size=batch_size,
          callbacks=[early_stopping, mcp_save],
          validation_data=(val_images, val_labels),
          epochs=epochs)
model.load_weights("best.mdl_wts_cifar100.hdf5")
test_loss, test_acc, b = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTop5 accuracy:', b)
