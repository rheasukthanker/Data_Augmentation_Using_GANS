import numpy as np
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar100 import load_data
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Dense, Flatten, Reshape, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from matplotlib import pyplot
from layers import ConvMaxout
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf


def binary_crossentropy_masked(y_true, y_pred):
    #define masked objective (mask real/fake loss for real samples)
    y_true_masked = tf.boolean_mask(y_true, tf.equal(y_true, 1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.equal(y_true, 1))
    return K.mean(K.binary_crossentropy(y_true_masked, y_pred_masked))


def generate_latent_representations(num_samples, real_samples, real_labels,
                                    perc_real, n_classes):
    #generate latent representations (mixture of real and fake)
    num_real = int(perc_real * num_samples)
    num_fake = num_samples - num_real
    real_inds = np.random.randint(0, real_samples.shape[0], size=num_real)
    real_z = real_samples[real_inds, :]
    real_labels = real_labels[real_inds]
    fake_z = np.random.normal(size=(num_fake, 1, 1, 64))
    fake_labels = np.random.randint(0, n_classes, num_fake)
    real_labels = np.reshape(real_labels, [np.shape(real_labels)[0], 1])
    fake_labels = np.reshape(fake_labels, [np.shape(fake_labels)[0], 1])
    all_z = np.concatenate([real_z, fake_z], axis=0)
    all_labels = np.vstack([real_labels, fake_labels])
    return [all_z, all_labels]


def generate_latent_points(latent_dim, n_samples, n_classes=2):
    # generate points in the latent space
    x_input = np.random.normal(size=(n_samples, 1, 1, latent_dim))
    # reshape into a batch of inputs for the network
    z_input = x_input
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_imbalance(trainX, trainy, class_cur, perc):
    class_to_sparsify = class_cur
    inds_class = np.argwhere(trainy == class_to_sparsify)
    num_delete = int(perc * len(inds_class))
    index = np.random.choice(inds_class.shape[0], num_delete, replace=False)
    index_class = inds_class[index]
    trainX_sparsified = np.delete(trainX, index_class, axis=0)
    trainy_sparsified = np.delete(trainy, index_class)
    return trainX_sparsified, trainy_sparsified


def generate_latent_points(latent_dim, n_samples, n_classes=2):
    # generate points in the latent space
    x_input = np.random.normal(size=(n_samples, 1, 1, latent_dim))
    # reshape into a batch of inputs for the network
    z_input = x_input
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# define the standalone discriminator model
def define_discriminator(in_shape=(50, 50, 3), n_classes=2):
    input_x = Input(shape=in_shape)
    x = Dropout(0.2)(input_x)
    x = Conv2D(32, (5, 5), strides=(1, 1))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4, 4), strides=(2, 2))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (4, 4), strides=(1, 1))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (4, 4), strides=(2, 2))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (4, 4), strides=(1, 1))(x)
    x = Conv2D(512, (5, 5), strides=(1, 1))(x)
    x = ConvMaxout(n_piece=2)(x)
    c1 = Dropout(0.5)(x)
    c1 = Conv2D(1024, (1, 1), strides=(1, 1))(c1)
    c1 = ConvMaxout(n_piece=2)(c1)
    c1 = Dropout(0.5)(c1)
    c1 = Conv2D(1024, (1, 1), strides=(1, 1))(c1)
    c1 = ConvMaxout(n_piece=2)(c1)
    c1 = Dropout(0.5)(c1)
    out1 = Conv2D(1, (1, 1), strides=(1, 1))(c1)
    out1 = Dense(1, activation="sigmoid")(Flatten()(out1))
    c2 = Dropout(0.5)(x)
    c2 = Conv2D(1024, (1, 1), strides=(1, 1))(c2)
    c2 = ConvMaxout(n_piece=2)(c2)
    c2 = Dropout(0.5)(c2)
    c2 = Conv2D(1024, (1, 1), strides=(1, 1))(c2)
    c2 = ConvMaxout(n_piece=2)(c2)
    c2 = Dropout(0.5)(c2)
    out2 = Conv2D(10, (1, 1), strides=(1, 1))(c2)
    out2 = Dense(n_classes, activation="softmax")(Flatten()(out2))
    # define model
    model = Model(input_x, [out1, out2])
    # compile model
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
    input = Input(shape=(1, 1, 64))
    label = Input(shape=(1, ))
    class_embs = Embedding(n_classes, 64, input_length=1)(label)
    input2 = Reshape((64, ))(input)
    class_embs2 = Reshape((64, ))(class_embs)
    input_concat = Concatenate(axis=1)([input2, class_embs2])
    x = Dense(64, input_shape=(128, ))(input_concat)
    x = Reshape((1, 1, 64))(x)
    x = Conv2DTranspose(256, (5, 5), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(128, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(64, (7, 7), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(32, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(32, (5, 5), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, (2, 2), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    output = Conv2D(3, (1, 1), strides=(1, 1), activation='sigmoid')(x)
    return Model([input, label], output)


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5)
    model.compile(
        loss=[binary_crossentropy_masked, 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


def load_real_samples():
    #load dataset
    with open("../../data/all_images.pkl", "rb") as f:
        all_images = pickle.load(f)
    with open("../../data/all_labels.pkl", "rb") as f:
        all_labels = pickle.load(f)
    randomize = np.arange(np.shape(all_images)[0])
    np.random.RandomState(seed=42).shuffle(randomize)
    X = all_images[randomize, :, :, :]
    y = all_labels[randomize]
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)
    del all_images
    del all_labels
    X = x_train.astype('float32')
    X = X / 255.
    y = y_train
    return [X, y]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, :])
    # save plot to file
    pyplot.show()


# train the generator and discriminator
def train(g_model,
          d_model,
          gan_model,
          dataset,
          latent_dim,
          n_epochs=300,
          n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    #load real z obtained from ALI Encoder
    with open('../../data/z_train_med.pkl', 'rb') as f:
        real_samples = pickle.load(f)
    with open('../../data/z_train_labels_med.pkl', 'rb') as f:
        real_labels = pickle.load(f)
    # manually enumerate epochs
    lossg = []
    lossd_real = []
    lossd_fake = []
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real,
             labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            _, d_r1, d_r2 = d_model.train_on_batch(X_real,
                                                   [y_real, labels_real])
            # generate 'fake' examples
            [X_fake, labels_fake
             ], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            _, d_f, d_f2 = d_model.train_on_batch(X_fake,
                                                  [y_fake, labels_fake])
            # prepare points in latent space as input for the generator
            [z_input, z_labels
             ] = generate_latent_representations(n_batch, real_samples,
                                                 real_labels, 0.25, 2)
            # create inverted labels for the fake samples,i.e, we pass the fake images generated to the gan model as real images. So, the loss is high if the
            # discriminator correctly classifies the image as fake. Since loss has to be minimized, the generator has to produce images so that the discriminator
            # classifies them as real.
            inds_shuffle = np.random.permutation(range(0, z_input.shape[0]))
            z_input = z_input[inds_shuffle, :]
            z_labels = z_labels[inds_shuffle]
            n_real = int(0.25 * n_batch)
            y_real = np.zeros((n_real, 1))
            y_fake = np.ones((n_batch - n_real, 1))
            y_gan = np.concatenate([y_real, y_fake], axis=0)
            y_gan = y_gan[inds_shuffle, :]
            # update the generator via the discriminator's error
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels],
                                                   [y_gan, z_labels])
            # summarize loss on this batch
            lossg.append([g_1, g_2])
            lossd_fake.append([d_f, d_f2])
            lossd_real.append([d_r1, d_r2])
            # evaluate the model performance every 'epoch'
            print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' %
                  ((i + 1) * (j + 1), d_r1, d_r2, d_f, d_f2, g_1, g_2))
            # print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
        filename = 'acgan_med_model_' + str(i) + '.h5'
        g_model.save(filename)
    with open("generator_loss_med_test3.pkl", "wb") as f:
        pickle.dump(lossg, f)
    with open("discriminator_real_loss_med_test3.pkl", "wb") as f:
        pickle.dump(lossd_real, f)
    with open("discriminator_fake_loss_med_test3.pkl", "wb") as f:
        pickle.dump(lossd_fake, f)


# size of the latent space
latent_dim = 64

# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
#load pretrained generator architecture same as ALI generator
generator.load_weights('../../data/Med_ALI_weights/xgen_weights.h5')
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
