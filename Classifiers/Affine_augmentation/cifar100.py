from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
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
epochs = 100
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS=3
NUM_CLASSES=100
latent_dim=64
num_samples=20
model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('elu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('elu'))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy','sparse_top_k_categorical_accuracy'])

#CIFAR 100
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_cifar100_2.hdf5', save_best_only=True, monitor='val_loss', mode='min')
cifar100=tf.keras.datasets.cifar100

(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
train_images = train_images/ 255.0
test_images = test_images/255.0

classes_to_sparsify=[83,69,35,34,2,90,37,49,23,26,41,62,33,12,84,50,57,74,93,40]
sparsify_percentage=[0.65,0.65,0.65,0.65,0.65,0.7,0.7,0.7,0.7,0.7,0.75,0.75,0.75,0.75,0.75,0.8,0.8,0.8,0.8,0.8]
for i in range(len(classes_to_sparsify)):
    train_images,train_labels=generate_imbalance(train_images,train_labels,classes_to_sparsify[i],sparsify_percentage[i])

train_images,val_images,train_labels,val_labels = train_test_split(train_images,train_labels, test_size=0.1)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,
                 width_shift_range=0.1, height_shift_range=0.1,
                 horizontal_flip=True)
datagen.fit(train_images)

class_weights = class_weight.compute_class_weight('balanced',np.unique(train_labels),np.squeeze(train_labels))
model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),class_weight=class_weights,callbacks=[early_stopping, mcp_save],validation_data=(val_images,val_labels), epochs=epochs)
model.load_weights("best.mdl_wts_cifar100_2.hdf5")
test_loss, test_acc,c= model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:', test_acc)
labels=model.predict(test_images)
print("Confusion matrix",confusion_matrix(test_labels.argmax(axis=1), labels.argmax(axis=1)))
