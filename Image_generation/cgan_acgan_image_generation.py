import numpy as np
import argparse
from PIL import Image
from tensorflow.keras.models import load_model

def decode_output(x):
    x = x.astype(np.float32)
    x *= 255.
    return x

def generate_latent_points(latent_dim, n_samples, n_classes):
	"""
	generate latent points and random labels
	"""
	x_input = np.random.randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	labels = np.random.randint(0, n_classes, n_samples)
	return [z_input, labels]


def generate_img(model_name,latent_dim,n_classes):
    num_generate_imgs = 25
    model = load_model(model_name)
    latent_points, labels = generate_latent_points(latent_dim,num_generate_imgs,n_classes)
    x_gen = model.predict([latent_points, labels])
    x_gen=(x_gen+1)/2

    x_gen = decode_output(x_gen)
    x_gen = np.clip(x_gen, 0., 255.).astype(np.uint8)

    # Concatenate generated images
    grid_size = int(np.sqrt(num_generate_imgs))
    rows = []
    for i in range(0, num_generate_imgs, grid_size):
        row = np.concatenate(x_gen[i:i+grid_size], axis=1)
        rows.append(row)
    concatenated = np.concatenate(rows, axis=0)
    return Image.fromarray(np.squeeze(concatenated))


parser = argparse.ArgumentParser(description='Image generation')
parser.add_argument('--model_name', nargs="?", type=str, default='../data/ACGAN_base/acgan_generator_fashion_mnist.h5', help='model_name')
parser.add_argument('--latent_dim', nargs="?", type=int, default=64, help='latent_dim=100 for CGAN, latent_dim=64 for ACGAN')
parser.add_argument('--n_classes', nargs="?", type=int, default=10, help='num of classes in dataset')
parser.add_argument('--image_name', nargs="?", type=str, default='generated_fashion_mnist.png', help='image name with format' )
args = parser.parse_args()
model_name=args.model_name
latent_dim=args.latent_dim
n_classes = args.n_classes
image_name=args.image_name

img = generate_img(str(model_name),latent_dim,n_classes)
img.save(str(image_name))
