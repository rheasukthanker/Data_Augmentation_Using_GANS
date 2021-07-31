import keras
from keras import layers
from keras.datasets.fashion_mnist import load_data
import numpy as np


#Sparsify 1,3,7 (50%,65%,80%)
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y


def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    # label input
    in_label = keras.Input(shape=(1, ))
    # embedding for categorical input
    li = layers.Embedding(n_classes, 100)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = layers.Dense(n_nodes)(li)
    # reshape to additional channel
    li = layers.Reshape((in_shape[0], in_shape[1], 1))(li)
    # image input
    in_image = keras.Input(shape=in_shape)
    # concat label as a channel
    merge = layers.Concatenate()([in_image, li])
    # downsample
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = layers.Flatten()(fe)
    # dropout
    fe = layers.Dropout(0.4)(fe)
    # output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # define model
    model = keras.Model([in_image, in_label], out_layer)
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def define_generator(latent_dim, n_classes=10):
    # label input
    in_label = keras.Input(shape=(1, ))
    # embedding for categorical input
    li = layers.Embedding(n_classes, 100)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = layers.Dense(n_nodes)(li)
    # reshape to additional channel
    li = layers.Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = keras.Input(shape=(latent_dim, ))
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((7, 7, 128))(gen)
    # merge image gen and label input
    merge = layers.Concatenate()([gen, li])
    # upsample to 14x14
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                 padding='same')(merge)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2),
                                 padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = layers.Conv2D(1, (7, 7), activation='tanh',
                              padding='same')(gen)
    # define model
    model = keras.Model([in_lat, in_label], out_layer)
    return model


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = keras.Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_real_samples(classes, percs):
    #load dataset
    (trainX, trainy), (testX, testy) = load_data()
    for i in range(len(classes)):
        trainX, trainy = generate_imbalance(trainX, trainy, classes[i],
                                            percs[i])
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5  #Range [-1,1]
    return [X, trainy]


def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]


def train(g_model,
          d_model,
          gan_model,
          dataset,
          latent_dim,
          n_epochs=300,
          n_batch=64):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real,
             labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake,
             labels], y_fake = generate_fake_samples(g_model, latent_dim,
                                                     half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input,
             labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        g_model.save('cgan_generator_fashion_mnist_epoch' + str(i + 1) + '.h5')


latent_dim = 100
classes_to_sparsify = [1, 3, 7]  # Define 3 random classes to sparsify
sparsify_percentage = [0.5, 0.65, 0.8]  #Percentage of classes to keep
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples(classes_to_sparsify, sparsify_percentage)
train(g_model, d_model, gan_model, dataset, latent_dim)
