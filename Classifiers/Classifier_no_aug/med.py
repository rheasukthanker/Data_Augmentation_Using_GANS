#Note same as full dataset as this dataset is naturally sparsified (no need for artificial sparsifying)
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import pickle
batch_size = 64
epochs = 100
IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_CHANNELS=3
NUM_CLASSES=2
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
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts_med.hdf5', save_best_only=True, monitor='val_loss', mode='min')
with open("../../data/all_images.pkl","rb") as f:
    all_images=pickle.load(f)
with open("../../data/all_labels.pkl","rb") as f:
    all_labels=pickle.load(f)
randomize = np.arange(np.shape(all_images)[0])
np.random.RandomState(seed=42).shuffle(randomize)
X = all_images[randomize,:,:,:]
y = all_labels[randomize]
del all_images
del all_labels
train_images,test_images,train_labels,test_labels = train_test_split(X,y, test_size=0.25, random_state=42)
train_images = train_images / 255.0
test_images = test_images / 255.0
class_weights = class_weight.compute_class_weight('balanced',np.unique(np.squeeze(train_labels)),np.squeeze(train_labels))
model.fit(train_images, train_labels,batch_size=batch_size,callbacks=[early_stopping, mcp_save],validation_split=0.1,class_weight=class_weights,epochs=epochs)
model.load_weights("best.mdl_wts_med.hdf5")
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
labels=model.predict(test_images)
print("Confusion matrix",confusion_matrix(test_labels, labels.argmax(axis=1)))
print('\nTest accuracy:', test_acc)
