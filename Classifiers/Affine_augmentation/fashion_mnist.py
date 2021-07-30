from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.utils import class_weight
def top2_acc(labels, logits):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=2)
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
latent_dim=64
num_samples=10000
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
#FASHION MNIST
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_fashion_mnist.hdf5', save_best_only=True, monitor='val_loss', mode='min')
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Normalize data
train_images =train_images/255.
test_images =test_images/255.
#Fit model
classes_to_sparsify=[1,3,7]# Define 3 random classes to sparsify
sparsify_percentage=[0.5,0.65,0.8] #Amount by which classes are sparsified
for i in range(len(classes_to_sparsify)):
    train_images,train_labels=generate_imbalance(train_images,train_labels,classes_to_sparsify[i],sparsify_percentage[i])
train_images=np.expand_dims(train_images,3)
test_images=np.expand_dims(test_images,3)
train_images,val_images,train_labels,val_labels = train_test_split(train_images,train_labels, test_size=0.1)
#define data augmentor
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.3,
            zoom_range=(0.9, 1.1),
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='constant',
            cval=0)
datagen.fit(train_images)

class_weights = class_weight.compute_class_weight('balanced',np.unique(np.squeeze(train_labels)),np.squeeze(train_labels))
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),callbacks=[early_stopping, mcp_save],class_weight=class_weights,validation_data=(val_images,val_labels), epochs=epochs)
model.load_weights("best.mdl_wts_fashion_mnist.hdf5")
#Get test accuracy
test_loss, test_acc,c= model.evaluate(test_images,test_labels,verbose=2)
labels=model.predict(test_images)
print("Confusion matrix",confusion_matrix(test_labels, labels.argmax(axis=1)))
print('\nTest accuracy:', test_acc)
