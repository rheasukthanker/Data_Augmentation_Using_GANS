#ACGAN on CIFAR100 with generator architecture same as generator of BiGAN and discriminator similar to discriminator of BiGAN
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.datasets.cifar100 import load_data
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Dense, Flatten, Reshape, Dropout
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from layers import ConvMaxout


def generate_imbalance(trainX, trainy, class_cur, perc):
    """
    sparcify the provided classes
    """
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


def generate_fake_samples(generator, latent_dim, n_samples):
    """
    Pass noise and label to generator to obtain fake images
    """
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


def generate_latent_points(latent_dim, n_samples, n_classes=100):
    """
    generate latent points and random labels
    """
    z_input = np.random.normal(size=(n_samples, 1, 1, latent_dim))
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def load_real_samples(classes, percs):
    """
    Load dataset
    """
    (trainX, trainy), (testX, testy) = load_data()
    for i in range(len(classes)):
        trainX, trainy = generate_imbalance(trainX, trainy, classes[i],
                                            percs[i])
    X = trainX.astype('float32')
    return [X, trainy]


def generate_real_samples(dataset, n_samples):
    """
    Sample images from dataset
    """
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    X = X / 255.
    y = ones((n_samples, 1))
    return [X, labels], y


def define_discriminator(in_shape=(32, 32, 3), n_classes=100):
    """
    Discriminator model
    """
    in_image = Input(shape=in_shape)
    x = Dropout(0.2)(in_image)
    x = Conv2D(32, (1, 1), strides=(1, 1))(x)
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
    x = ConvMaxout(n_piece=2)(x)
    c = Dropout(0.5)(x)
    c = Conv2D(1024, (1, 1), strides=(1, 1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(1024, (1, 1), strides=(1, 1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    out1 = Conv2D(1, (1, 1), strides=(1, 1))(c)
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
    model = Model(in_image, [out1, out2])
    # compile model
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


def define_generator():
    """
    Generator model
    """
    input = Input(shape=(1, 1, 64))
    label = Input(shape=(1, ))
    class_embs = Embedding(100, 64, input_length=1)(label)
    input2 = Reshape((64, ))(input)
    print(class_embs.shape)
    class_embs2 = Reshape((64, ))(class_embs)
    input_concat = Concatenate(axis=1)([input2, class_embs2])
    x = Dense(64, input_shape=(128, ))(input_concat)
    x = Reshape((1, 1, 64))(x)
    x = Conv2DTranspose(256, (4, 4), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(64, (4, 4), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2DTranspose(32, (5, 5), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, (1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    output = Conv2D(3, (1, 1), strides=(1, 1), activation='sigmoid')(x)
    return Model([input, label], output)


def define_gan(g_model, d_model):
    """
    GAN model
    """
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


def train(g_model,
          d_model,
          gan_model,
          dataset,
          latent_dim,
          n_epochs=300,
          n_batch=64):
    """
    Training GAN model
    """
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            #train discriminator
            [X_real,
             labels_real], y_real = generate_real_samples(dataset, half_batch)
            _, d_r1, d_r2 = d_model.train_on_batch(X_real,
                                                   [y_real, labels_real])

            [X_fake, labels_fake
             ], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            _, d_f, d_f2 = d_model.train_on_batch(X_fake,
                                                  [y_fake, labels_fake])
            # train generator
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels],
                                                   [y_gan, z_labels])
            # summarize loss on this batch
            print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' %
                  (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        if ((i + 1) % 2) == 0:
            g_model.save('acgan_generator_arch2_cifar100_epoch' + str(i + 1) +
                         '.h5')  #model saved every seconnd epoch


# size of the latent space
latent_dim = 64
classes_to_sparsify = [
    83, 69, 35, 34, 2, 90, 37, 49, 23, 26, 41, 62, 33, 12, 84, 50, 57, 74, 93,
    40
]
sparsify_percentage = [
    0.65, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8
]

# create the models
discriminator = define_discriminator()
generator = define_generator()
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples(classes_to_sparsify, sparsify_percentage)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
