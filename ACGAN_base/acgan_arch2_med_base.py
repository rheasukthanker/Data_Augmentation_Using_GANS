#ACGAN on medical data with generator architecture same as generator of BiGAN and discriminator similar to discriminator of BiGAN 
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Dense, Flatten, Reshape, Dropout
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from layers import ConvMaxout
import pickle
from sklearn.model_selection import train_test_split

    
def generate_latent_points(latent_dim, n_samples, n_classes=2):
    """
    generate latent points and random labels
    """
    z_input =np.random.normal(size=(n_samples,1,1,latent_dim))
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    """
    Pass noise and label to generator to obtain fake images
    """
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [images, labels_input], y


def load_real_samples():
	"""
	Load dataset
	"""
	with open("../data/all_images.pkl", "rb") as f:
		all_images = pickle.load(f)
	with open("../data/all_labels.pkl", "rb") as f:
		all_labels = pickle.load(f)
	randomize = np.arange(np.shape(all_images)[0])
	np.random.RandomState(seed=42).shuffle(randomize)
	X = all_images[randomize, :, :, :]
	y = all_labels[randomize]
	x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	del all_images
	del all_labels
	X =x_train.astype('float32')
	X = X / 255.
	y=y_train
	return [X,y]


def generate_real_samples(dataset, n_samples):
    """
    Sample images from dataset
    """
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y


def define_discriminator(in_shape=(50, 50, 3), n_classes=2):
    """
    Discriminator model
    """
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
    c1= Dropout(0.5)(x)
    c1= Conv2D(1024, (1, 1), strides=(1, 1))(c1)
    c1= ConvMaxout(n_piece=2)(c1)
    c1= Dropout(0.5)(c1)
    c1= Conv2D(1024, (1, 1), strides=(1, 1))(c1)
    c1= ConvMaxout(n_piece=2)(c1)
    c1= Dropout(0.5)(c1)
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
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


def define_generator(latent_dim, n_classes=2):
    """
    Generator model
    """
    input = Input(shape=(1, 1, 64))
    label = Input(shape=(1,))
    class_embs = Embedding(n_classes, 64, input_length=1)(label)
    input2 = Reshape((64,))(input)
    class_embs2 = Reshape((64,))(class_embs)
    input_concat = Concatenate(axis=1)([input2, class_embs2])
    x = Dense(64, input_shape=(128,))(input_concat)
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
    return Model([input,label], output)



def define_gan(g_model, d_model):
    """
    GAN model
    """
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = optimizers.Adam(lr=0.00001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model





def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300, n_batch=64):
    """
    Training GAN model
    """
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
       for j in range(bat_per_epo):
        # train discriminator
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])

        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # train generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
       if ((i+1)%2)==0:
        g_model.save('acgan_generator_arch2_med_epoch'+str(i+1)+'.h5') #model saved every second epoch



# size of the latent space
latent_dim = 64

# create the models
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)

