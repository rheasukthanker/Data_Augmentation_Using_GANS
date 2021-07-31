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


def generate_latent_representations(num_samples, real_samples, real_labels,
                                    perc_real, n_classes):
    #generate latent points mixture of real and fake
    num_real = int(perc_real * num_samples)
    num_fake = num_samples - num_real
    real_inds = np.random.randint(0, real_samples.shape[0], size=num_real)
    real_z = real_samples[real_inds, :]
    real_labels = real_labels[real_inds]
    fake_z = np.random.normal(size=(num_fake, 1, 1, 64))
    fake_labels = np.random.randint(0, n_classes, num_fake)
    fake_labels = np.expand_dims(fake_labels, 1)
    all_z = np.concatenate([real_z, fake_z], axis=0)
    real_labels = np.reshape(real_labels, [np.shape(real_labels)[0], 1])
    fake_labels = np.reshape(fake_labels, [np.shape(fake_labels)[0], 1])
    all_labels = np.vstack([real_labels, fake_labels])
    inds_shuffle = np.random.permutation(range(0, all_z.shape[0]))
    all_z = all_z[inds_shuffle, :]
    all_labels = all_labels[inds_shuffle]
    return [all_z, all_labels]


def generate_imbalance(trainX, trainy, class_cur, perc):
    #create imbalance
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


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


def generate_latent_points(latent_dim, n_samples, n_classes=100):
    # generate points in the latent space latent/fake z
    x_input = np.random.normal(size=(n_samples, 1, 1, latent_dim))
    # reshape into a batch of inputs for the network
    z_input = x_input
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3), n_classes=100):
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


# define the standalone generator model
def define_generator():
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
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
        optimizer=opt)
    return model


def load_real_samples(classes, percs):
    (trainX, trainy), (testX, testy) = load_data()
    for i in range(len(classes)):
        trainX, trainy = generate_imbalance(trainX, trainy, classes[i],
                                            percs[i])
    X = trainX.astype('float32')
    return [X, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    X = X / 255.
    y = ones((n_samples, 1))
    return [X, labels], y


def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
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
          n_epochs=200,
          n_batch=64):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    #load real z obtained from pretrained ALI encoder
    with open('../../data/z_train_cifar100.pkl', 'rb') as f:
        real_samples = pickle.load(f)
    with open('../../data/z_train_labels_cifar100.pkl', 'rb') as f:
        real_labels = pickle.load(f)
    num_samples = n_batch
    # manually enumerate epochs
    classes_to_sparsify = [
        83, 69, 35, 34, 2, 90, 37, 49, 23, 26, 41, 62, 33, 12, 84, 50, 57, 74,
        93, 40
    ]
    sparsify_percentage = [
        0.65, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.75, 0.75,
        0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8
    ]
    for i in range(len(classes_to_sparsify)):
        real_samples, real_labels = generate_imbalance(real_samples,
                                                       real_labels,
                                                       classes_to_sparsify[i],
                                                       sparsify_percentage[i])
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
             ] = generate_latent_representations(num_samples, real_samples,
                                                 real_labels, 0.75, 100)
            # create inverted labels for the fake samples,i.e, we pass the fake images generated to the gan model as real images. So, the loss is high if the
            # discriminator correctly classifies the image as fake. Since loss has to be minimized, the generator has to produce images so that the discriminator
            # classifies them as real.
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels],
                                                   [y_gan, z_labels])
            # summarize loss on this batch
            # summarize loss on this batch
            lossg.append([g_1, g_2])
            lossd_fake.append([d_f, d_f2])
            lossd_real.append([d_r1, d_r2])
            # evaluate the model performance every 'epoch'
            print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' %
                  ((i + 1) * (j + 1), d_r1, d_r2, d_f, d_f2, g_1, g_2))
        filename = 'acgan_cifar100_model_' + str(i) + '.h5'
        g_model.save(filename)
    with open("generator_loss_cifar_test1.pkl", "wb") as f:
        pickle.dump(lossg, f)
    with open("discriminator_real_loss_cifar_test1.pkl", "wb") as f:
        pickle.dump(lossd_real, f)
    with open("discriminator_fake_loss_cifar_test1.pkl", "wb") as f:
        pickle.dump(lossd_fake, f)


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

# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator()
#load generator pretrained from ALI (pretrianed)
generator.load_weights('../../data/CIFAR_ALI_weights/xgen_weights.h5')
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples(classes_to_sparsify, sparsify_percentage)
# dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
