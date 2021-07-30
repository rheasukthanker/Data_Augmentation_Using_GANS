#Classifier trained on augmented data of fashion MNIST
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import argparse
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import load_model
from numpy.random import randn
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.metrics import sparse_top_k_categorical_accuracy



def top2_acc(labels, logits):
    return sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=2)

def get_images_from_generator(label,model_name,num_samples,latent_dim):
	model = load_model(model_name)
	latent_points, labels = generate_latent_points(latent_dim,num_samples,label)
	X = model.predict([latent_points, labels])
	return [X,labels]

def generate_latent_points(latent_dim, num_samples,label):
	x_input = randn(latent_dim * num_samples)
	z_input = x_input.reshape(num_samples, latent_dim)
	labels = np.repeat(label,num_samples)
	return [z_input, labels]

def generate_imbalance(trainX,trainy,class_cur,perc):
	class_to_sparsify=class_cur
	inds_class=np.argwhere(trainy==class_to_sparsify)
	num_delete=int((1-perc)*len(inds_class))
	index = np.random.RandomState(seed=42).choice(inds_class.shape[0], num_delete, replace=False)
	index_class=inds_class[index]
	trainX_sparsified=np.delete(trainX,index_class,axis=0)
	trainy_sparsified=np.delete(trainy,index_class)
	return trainX_sparsified,trainy_sparsified

#Obtaining arguments
parser = argparse.ArgumentParser(description='Classifier with GAN augmentation')
parser.add_argument('--model_name', nargs="?", type=str, default='../../data/ACGAN_base/acgan_generator_fashion_mnist.h5', help='model_name')
parser.add_argument('--num_epoch', nargs="?", type=int, default=100, help='number of epochs')
args = parser.parse_args()
model_name=args.model_name
epochs = args.num_epoch

batch_size = 64
IMG_HEIGHT =28
IMG_WIDTH = 28
NUM_CHANNELS=1
NUM_CLASSES=10
latent_dim=64


#classifier model
model= Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,NUM_CHANNELS)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',top2_acc])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_fashion_mnist.hdf5', save_best_only=True, monitor='val_loss', mode='min')


#Load real data and sparsify to match set on which GAN was trained
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Normalize data
train_images= (train_images - 127.5) / 127.5
test_images = (test_images- 127.5) / 127.5
classes_to_sparsify=[1,3,7]# Define 3 random classes to sparsify
sparsify_percentage=[0.5,0.65,0.8] #Amount by which classes are sparsified
for i in range(len(classes_to_sparsify)):
    train_images,train_labels=generate_imbalance(train_images,train_labels,classes_to_sparsify[i],sparsify_percentage[i])
train_images=np.expand_dims(train_images,3)
test_images=np.expand_dims(test_images,3)
train_images,val_images,train_labels,val_labels = train_test_split(train_images,train_labels, test_size=0.1)

#Generate and augment data to create balanced dataset. Images generated from model mentioned in arguments
gan_samples=[]
gan_labels=[]
for i in range(len(classes_to_sparsify)):
        [X,labels]=get_images_from_generator(classes_to_sparsify[i],str(model_name),int((1-sparsify_percentage[i])*6000),latent_dim)
        gan_samples.append(X)
        gan_labels.append(np.expand_dims(labels,1))
#append images
gan_samples.append(train_images)
gan_labels.append(np.expand_dims(train_labels,1))
gan_samples=np.vstack(gan_samples)
gan_labels=np.vstack(gan_labels)

inds=np.random.permutation(gan_samples.shape[0])
gan_samples=gan_samples[inds,:,:,:]
gan_labels=gan_labels[inds]
class_weights = class_weight.compute_class_weight('balanced',np.unique(np.squeeze(gan_labels)),np.squeeze(gan_labels))
model.fit(gan_samples, gan_labels,batch_size=batch_size,callbacks=[early_stopping, mcp_save],validation_data=(val_images,val_labels),class_weight=class_weights, epochs=epochs)
model.load_weights("best.mdl_wts_fashion_mnist.hdf5")
#Get test accuracy
test_loss, test_acc,acc2 = model.evaluate(test_images,test_labels,verbose=2)
predict_prob=model.predict(test_images)
y_pred = np.argmax(predict_prob, axis=1)
cmatrix=confusion_matrix(np.squeeze(test_labels), np.squeeze(y_pred))
print('\n Confusion Matrix')
print(cmatrix)
print('\nTest accuracy :', test_acc)
print('\nTop2 accuracy :', acc2)
