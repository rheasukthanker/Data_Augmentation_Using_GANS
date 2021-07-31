import keras.backend as K
from keras.optimizers import Adam
from keras.utils import Progbar
import numpy as np
import models
import utils
import os
import argparse
from sklearn.model_selection import train_test_split
import pickle

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--beta_2', type=float, default=0.999)
parser.add_argument('--snap_freq', type=int, default=5)
parser.add_argument('--result', default=os.path.join(curdir, 'result'))


def save_config(path, args):
    with open(path, 'w') as f:
        f.write('Epochs: %d\n' % (args.epochs))
        f.write('Batchsize: %d\n' % (args.batch_size))
        f.write('Learning rate: %f\n' % (args.lr))
        f.write('Beta_1: %f\n' % (args.beta_1))
        f.write('Beta_2: %f\n' % (args.beta_2))


def d_lossfun(y_true, y_pred):
    """
    y_pred[:,:,:,0]: p
    y_pred[:,:,:,1]: q
    """
    p = K.clip(y_pred[:, :, :, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, :, :, 1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(q) + K.log(1. - p))


def g_lossfun(y_true, y_pred):
    """
    y_pred[:,:,:,0]: p
    y_pred[:,:,:,1]: q
    """
    p = K.clip(y_pred[:, :, :, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, :, :, 1], K.epsilon(), 1.0 - K.epsilon())
    return -K.mean(K.log(1. - q) + K.log(p))


def main(args):

    # =====================================
    # Preparation (load dataset and create
    # a directory which saves results)
    # =====================================
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
    #x_train=X
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)
    x_train = utils.preprocess_input(x_train)
    if os.path.exists(args.result) == False:
        os.makedirs(args.result)
    save_config(os.path.join(args.result, 'config.txt'), args)

    # =====================================
    # Instantiate models
    # =====================================
    num_classes = 2
    input_shape = (50, 50, 3)
    xgen = models.create_xgenerater(num_classes)
    zgen = models.create_zgenerater(input_shape, num_classes)
    disc = models.create_discriminater(input_shape)
    opt_d = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)
    opt_g = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)

    xgen.trainable = False
    zgen.trainable = False
    gan_d = models.create_gan(xgen, zgen, disc)
    gan_d.compile(optimizer=opt_d, loss=d_lossfun)

    xgen.trainable = True
    zgen.trainable = True
    disc.trainable = False
    gan_g = models.create_gan(xgen, zgen, disc)
    gan_g.compile(optimizer=opt_g, loss=g_lossfun)

    # =====================================
    # Training Loop
    # =====================================
    num_train = len(x_train)
    for epoch in range(args.epochs):
        print('Epochs %d/%d' % (epoch + 1, args.epochs))
        pbar = Progbar(num_train)
        for i in range(0, num_train, args.batch_size):
            x = x_train[i:i + args.batch_size]
            z = np.random.normal(size=(len(x), 1, 1, 64))
            l = y_train[i:i + args.batch_size]
            print(max(l))
            # train discriminater
            d_loss = gan_d.train_on_batch([x, l, z], np.zeros(
                (len(x), 1, 1, 2)))
            # train generaters
            g_loss = gan_g.train_on_batch([x, l, z], np.zeros(
                (len(x), 1, 1, 2)))

            # update progress bar
            pbar.add(len(x), values=[
                ('d_loss', d_loss),
                ('g_loss', g_loss),
            ])

        if (epoch + 1) % args.snap_freq == 0:
            # ===========================================
            # Save result
            # ===========================================
            # Make a directory which stores learning results
            # at each (args.frequency)epochs
            dirname = 'epochs%d' % (epoch + 1)
            path = os.path.join(args.result, dirname)
            if os.path.exists(path) == False:
                os.makedirs(path)

            # Save generaters' weights
            xgen.save_weights(os.path.join(path, 'xgen_weights.h5'))
            zgen.save_weights(os.path.join(path, 'zgen_weights.h5'))

            # Save generated images
            img = utils.generate_img(y_test, xgen)
            img.save(os.path.join(path, 'generated.png'))

            # Save reconstructed images
            img = utils.reconstruct_img(x_test, y_test, xgen, zgen)
            img.save(os.path.join(path, 'reconstructed.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
