import numpy as np
import argparse
from PIL import Image
from tensorflow.keras.models import load_model


def decode_output(x):
    x = x.astype(np.float32)
    x *= 255.
    return x


def generate_img(model_name, n_classes):
    latent_dim = 64
    num_generate_imgs = 25
    model = load_model(model_name)
    labels = np.random.randint(0, n_classes, num_generate_imgs)

    z = np.random.normal(size=(num_generate_imgs, 1, 1, latent_dim))
    x_gen = model.predict([z, labels])
    x_gen = decode_output(x_gen)
    x_gen = np.clip(x_gen, 0., 255.).astype(np.uint8)

    # Concatenate generated images
    grid_size = int(np.sqrt(num_generate_imgs))
    rows = []
    for i in range(0, num_generate_imgs, grid_size):
        row = np.concatenate(x_gen[i:i + grid_size], axis=1)
        rows.append(row)
    concatenated = np.concatenate(rows, axis=0)
    return Image.fromarray(np.squeeze(concatenated))


parser = argparse.ArgumentParser(description='Image generation')
parser.add_argument(
    '--model_name',
    nargs="?",
    type=str,
    default=
    '../data/ACGAN_ALI_pretrained/V1/acgan_ali_v1_generator_fashion_mnist.h5',
    help='model_name')
parser.add_argument('--n_classes',
                    nargs="?",
                    type=int,
                    default=10,
                    help='num of classes in dataset')
parser.add_argument('--image_name',
                    nargs="?",
                    type=str,
                    default='generated_fashion_mnist.png',
                    help='image name with format')
args = parser.parse_args()
model_name = args.model_name
n_classes = args.n_classes
image_name = args.image_name

img = generate_img(str(model_name), n_classes)
img.save(str(image_name))
