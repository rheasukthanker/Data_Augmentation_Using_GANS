from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import load_model
from numpy.random import randn
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import itertools
from sklearn.preprocessing import normalize
from keras.utils import to_categorical
def top2_acc(labels, logits):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=2)
def get_images_from_generator(label,model_name,num_samples,latent_dim):
	model = load_model(model_name)
	latent_points, labels = generate_latent_points(latent_dim,num_samples,label)
	X = model.predict([latent_points, labels])
	return [X,labels]
def generate_latent_points(latent_dim, num_samples,label):
	# generate points in the latent space
	x_input = randn(latent_dim * num_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(num_samples, latent_dim)
	# generate labels
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
batch_size = 64
epochs =100
IMG_HEIGHT =28
IMG_WIDTH = 28
NUM_CHANNELS=1
NUM_CLASSES=10
latent_dim=100
num_samples=1000
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

adam = tf.keras.optimizers.Adam()
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',top2_acc])
#FASHION MNIST
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_fashion_mnist.hdf5', save_best_only=True, monitor='val_loss', mode='min')
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()#Change this line to modify train and test input
#Normalize data
train_images = (train_images - np.mean(train_images)) / np.std(train_images)
test_images = (test_images - np.mean(test_images)) / np.std(test_images)
#Fit model
classes_to_sparsify=[1,3,7]# Define 3 random classes to sparsify
sparsify_percentage=[0.5,0.65,0.8] #Amount by which classes are sparsified
for i in range(len(classes_to_sparsify)):
    train_images,train_labels=generate_imbalance(train_images,train_labels,classes_to_sparsify[i],sparsify_percentage[i])
train_images=np.expand_dims(train_images,3)
test_images=np.expand_dims(test_images,3)
train_images,val_images,train_labels,val_labels = train_test_split(train_images,train_labels, test_size=0.1)#, random_state=42)
gan_samples=[]
gan_labels=[]
weights=[]
for i in range(len(classes_to_sparsify)):
    [X,labels]=get_images_from_generator(classes_to_sparsify[i],"../../data/CGAN/cgan_generator_fashion_mnist.h5",int((1-sparsify_percentage[i])*6000),latent_dim)
    X = (X - np.mean(X)) / np.std(X)
    gan_samples.append(X)
    gan_labels.append(np.expand_dims(labels,1))
gan_samples.append(train_images)
gan_labels.append(np.expand_dims(train_labels,1))
gan_samples=np.vstack(gan_samples)
gan_labels=np.vstack(gan_labels)
inds=np.random.permutation(gan_samples.shape[0])
gan_samples=gan_samples[inds,:,:,:]
gan_labels=gan_labels[inds]
print(gan_samples.shape)
model.fit(gan_samples, gan_labels,batch_size=batch_size,callbacks=[early_stopping, mcp_save],validation_data=(val_images,val_labels), epochs=epochs)
model.load_weights("best.mdl_wts_fashion_mnist.hdf5")
#Get test accuracy
test_loss, test_acc,abc = model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:', test_acc)
