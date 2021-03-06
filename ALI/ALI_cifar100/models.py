import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Dense, Flatten, Reshape, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from layers import ConvMaxout


def create_xgenerater(n_classes):
    input = Input(shape=(1, 1, 64))
    label = Input(shape=(1, ))
    class_embs = Embedding(n_classes, 64, input_length=1)(label)
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
    return Model([input, label], output, name='xgenerater')


def create_zgenerater(input_shape, n_classes):
    input = Input(shape=input_shape)
    label = Input(shape=(1, ))
    x = Conv2D(32, (5, 5), strides=(1, 1))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (4, 4), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (4, 4), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (4, 4), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1, 1), strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    mu = Conv2D(64, (1, 1), strides=(1, 1))(x)
    class_embs = Embedding(n_classes, 64)(label)
    class_embs = Reshape((64, ))(class_embs)
    mu_sq = Reshape((64, ))(mu)
    input_concat_mu = Concatenate(axis=1)([mu_sq, class_embs])
    mu_final = Dense(64, input_shape=(128, ))(input_concat_mu)
    mu_final = Reshape((1, 1, 64))(mu_final)
    sigma = Conv2D(64, (1, 1), strides=(1, 1))(x)
    sigma_sq = Reshape((64, ))(sigma)
    input_concat_sigma = Concatenate(axis=1)([sigma_sq, class_embs])
    sigma_final = Dense(64, input_shape=(128, ))(input_concat_sigma)
    sigma_final = Reshape((1, 1, 64))(sigma_final)
    concatenated = Concatenate(axis=-1)([mu_final, sigma_final])

    output = Lambda(function=lambda x: x[:, :, :, :64] + K.exp(x[:, :, :, 64:])
                    * K.random_normal(shape=K.shape(x[:, :, :, 64:])),
                    output_shape=(1, 1, 64))(concatenated)
    return Model([input, label], output, name='zgenerater')


def create_discriminater(input_shape):
    input_x = Input(shape=input_shape)
    input_z = Input(shape=(1, 1, 64))

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
    x = ConvMaxout(n_piece=2)(x)

    z = Dropout(0.2)(input_z)
    z = Conv2D(512, (1, 1), strides=(1, 1))(z)
    z = ConvMaxout(n_piece=2)(z)
    z = Dropout(0.5)(z)
    z = Conv2D(512, (1, 1), strides=(1, 1))(z)
    z = ConvMaxout(n_piece=2)(z)

    concatenated = Concatenate(axis=-1)([x, z])
    c = Dropout(0.5)(concatenated)
    c = Conv2D(1024, (1, 1), strides=(1, 1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(1024, (1, 1), strides=(1, 1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid')(c)
    return Model([input_x, input_z], c, name='discriminater')


def create_gan(xgenerater, zgenerater, discriminater):
    input_shape = (32, 32, 3)
    input_x = Input(shape=input_shape)
    input_z = Input(shape=(1, 1, 64))
    labels_xz = Input(shape=(1, ))
    x_gen = xgenerater([input_z, labels_xz])
    z_gen = zgenerater([input_x, labels_xz])
    p = discriminater([x_gen, input_z])
    q = discriminater([input_x, z_gen])
    concatenated = Concatenate(axis=-1)([p, q])
    return Model([input_x, labels_xz, input_z], concatenated, name='gan')
