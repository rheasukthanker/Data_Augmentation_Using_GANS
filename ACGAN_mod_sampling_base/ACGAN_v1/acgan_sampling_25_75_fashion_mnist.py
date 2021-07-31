# Modified Auxiliary Classifier gan (ac-gan) on fashion mnsit with 25% real data and 75% fake data
import numpy as np
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal


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


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    """
	generate latent points and random labels
	"""
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
    """
	Pass noise and label to generator to obtain fake images
	"""
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))  #To indicate images are fake
    return [images, labels_input], y


def load_real_samples(classes, percs):
    """
	Load dataset
	"""
    (trainX, trainy), (testX, testy) = load_data()
    for i in range(len(classes)):
        trainX, trainy = generate_imbalance(trainX, trainy, classes[i],
                                            percs[i])
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainy]


def generate_real_samples(dataset, n_samples):
    """
	Sample images from dataset
	"""
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))  #To indicate images are real
    return [X, labels], y


def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    """
	Discriminator model
	"""
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    fe = Conv2D(32, (3, 3),
                strides=(2, 2),
                padding='same',
                kernel_initializer=init)(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    fe = Conv2D(128, (3, 3),
                strides=(2, 2),
                padding='same',
                kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    fe = Flatten()(fe)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    model = Model(in_image, [out1, out2])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


def define_generator(latent_dim, n_classes=10):
    """
	Generator model
	"""
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1, ))
    li = Embedding(n_classes, 64)(in_label)
    n_nodes = 7 * 7
    li = Dense(n_nodes, kernel_initializer=init)(li)
    li = Reshape((7, 7, 1))(li)
    in_lat = Input(shape=(latent_dim, ))
    n_nodes = 384 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((7, 7, 384))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(192, (5, 5),
                          strides=(2, 2),
                          padding='same',
                          kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    gen = Conv2DTranspose(1, (5, 5),
                          strides=(2, 2),
                          padding='same',
                          kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model


def define_gan(g_model, d_model):
    """
	GAN model
	"""
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
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
    quarter_batch = int(n_batch / 4)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # train discriminator with a batch containing 25% real data and 75% fake data
            [X_real, labels_real
             ], y_real = generate_real_samples(dataset, quarter_batch)
            _, d_r1, d_r2 = d_model.train_on_batch(X_real,
                                                   [y_real, labels_real])

            [X_fake, labels_fake
             ], y_fake = generate_fake_samples(g_model, latent_dim,
                                               3 * quarter_batch)
            _, d_f, d_f2 = d_model.train_on_batch(X_fake,
                                                  [y_fake, labels_fake])
            # train generator
            [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels],
                                                   [y_gan, z_labels])
            # summarize loss
            print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' %
                  (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
        if ((i + 1) % 2) == 0:
            g_model.save('acgan_generator_sampling_25_75_fashion_mnist_epoch' +
                         str(i + 1) + '.h5')


latent_dim = 64
# Clasees to sparsify with percentage
classes_to_sparsify = [1, 3, 7]
sparsify_percentage = [0.5, 0.65, 0.8]
# create the models
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples(classes_to_sparsify, sparsify_percentage)
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
