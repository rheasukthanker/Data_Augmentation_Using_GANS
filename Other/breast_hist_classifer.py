from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pickle
batch_size = 64
epochs = 20
IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_CHANNELS=3
NUM_CLASSES=2

def se_block(in_block, ch=32, ratio=4):  # check // ratio=16 // instead of plain residual connection
    x = tf.keras.layers.GlobalAveragePooling2D()(in_block)
    # x = Conv2D(ch/ratio,(1,1),(1,1),data_format="channels_first")(x)
    # x = tf.nn.swish(x)
    # x = Conv2D(ch,(1,1),(1,1),data_format="channels_first")(x)
    x = tf.keras.layers.Dense(ch // ratio)(x)
    x = tf.keras.layers.Activation(activation=swish)(x)
    x = tf.keras.layers.Dense(ch, activation='sigmoid')(x)

    return tf.keras.layers.multiply([in_block, x])#()

def swish(x):
    return tf.keras.layers.Multiply()([x, tf.keras.layers.Activation('sigmoid')(x)])

def MBConv(x, ksize=2, expand=64, squeeze=16):
    m = tf.keras.layers.Conv2D(expand, (1, 1), padding="same")(x)
    m = tf.keras.layers.BatchNormalization(axis=-1)(m)
    m = tf.keras.layers.Activation(activation=swish)(m)
    m = tf.keras.layers.DepthwiseConv2D([ksize, ksize], [1, 1], padding="same",depth_multiplier=2)(m)
    m = tf.keras.layers.BatchNormalization(axis=-1)(m)
    m = tf.keras.layers.Activation(activation=swish)(m)
    m = tf.keras.layers.Conv2D(squeeze, (1, 1), padding="same")(m)
    m = tf.keras.layers.BatchNormalization(axis=-1)(m)

    return tf.keras.layers.add([m, x])#()

def expand_dims(x):
    return tf.keras.backend.expand_dims(x, -1)


inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))#fashion mnist dimensions
#inp = tf.keras.layers.Lambda(expand_dims)(inputs)#tf.keras.backend.expand_dims(inputs,axis=-1)
conv_0_0 = tf.keras.layers.Conv2D(32, [2, 2], strides=[1, 1], padding="same")(
    inputs)  # to reduce dimension, can be left out for low dimension data #dimensionality reduction here #[2,2],[2,2], valid
conv_0_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_0_0)
conv_0_0 = tf.keras.layers.Activation(activation=swish)(conv_0_0)
conv_0_1 = tf.keras.layers.DepthwiseConv2D([2, 2], [1, 1], padding="same",depth_multiplier=1)(
    conv_0_0)  # check dimensions and regularizer
conv_0_1 = tf.keras.layers.BatchNormalization(axis=-1)(conv_0_1)
conv_0_1 = tf.keras.layers.Activation(activation=swish)(conv_0_1)
conv_0_2 = tf.keras.layers.DepthwiseConv2D([2, 2], [1, 1], padding="same",depth_multiplier=1)(conv_0_1)
conv_0_2 = tf.keras.layers.BatchNormalization(axis=-1)(conv_0_2)
conv_0_2 = tf.keras.layers.Activation(activation=swish)(conv_0_2)
se_1 = se_block(conv_0_2, ch=32, ratio=4)
mbc_0_0 = MBConv(se_1, 3, expand=98, squeeze=32)
conv_1_0 = tf.keras.layers.DepthwiseConv2D([2, 2], strides=[2, 2], padding="same",depth_multiplier=2)(
    mbc_0_0)
conv_1_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_1_0)
conv_1_0 = tf.keras.layers.Activation(activation=swish)(conv_1_0)
se_2 = se_block(conv_1_0, ch=64, ratio=4)
mbc_1_0 = MBConv(se_2, 3, expand=128, squeeze=64)
conv_2_0 = tf.keras.layers.DepthwiseConv2D([2, 2], strides=[1, 1], padding="same",depth_multiplier=2)(
    mbc_1_0)
conv_2_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_2_0)
conv_2_0 = tf.keras.layers.Activation(activation=swish)(conv_2_0)
se_3 = se_block(conv_2_0, ch=128, ratio=4)
mbc_2_0 = MBConv(se_3, 5, expand=256, squeeze=128)
se_4 = se_block(mbc_2_0, ch=128, ratio=4)
mbc_3_0 = MBConv(se_4, 5, expand=256, squeeze=128)
conv_3_0 = tf.keras.layers.DepthwiseConv2D([2, 2], strides=[1, 1], padding="same",depth_multiplier=2)(
    mbc_3_0)
conv_3_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_3_0)
conv_3_0 = tf.keras.layers.Activation(activation=swish)(conv_3_0)
se_5 = se_block(conv_3_0, ch=256, ratio=4)
mbc_4_0 = MBConv(se_5, 5, expand=512, squeeze=256)
se_6 = se_block(mbc_4_0, ch=256, ratio=4)
mbc_5_0 = MBConv(se_6, 5, expand=512, squeeze=256)
conv_4_0 = tf.keras.layers.DepthwiseConv2D([3, 3], strides=[2, 2], padding="valid",depth_multiplier=2)(
    mbc_5_0)
conv_4_0 = tf.keras.layers.BatchNormalization(axis=-1)(conv_4_0)
conv_4_0 = tf.keras.layers.Activation(activation=swish)(conv_4_0)
#conv_2_0 = tf.keras.layers.Conv2D(512, (1, 1), activation='relu')(conv_2_0)
conv_5_0 = tf.keras.layers.GlobalAveragePooling2D()(conv_4_0)
conv_5_0 = tf.keras.layers.Flatten()(conv_5_0)

outputs = tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')(conv_5_0)#,activation='softmax'
#outputs=conv_3_0
#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv_2_0)

model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Breast histo
with open("../data/all_images.pkl","rb") as f:
    all_images=pickle.load(f)
with open("../data/all_labels.pkl","rb") as f:
    all_labels=pickle.load(f)
randomize = np.arange(np.shape(all_images)[0])
np.random.shuffle(randomize)
X = all_images[randomize,:,:,:]
y = all_labels[randomize]
del all_images
del all_labels
train_images,test_images,train_labels,test_labels = train_test_split(X,y, test_size=0.25, random_state=42)
train_images = train_images / 255.0
test_images = test_images / 255.0
model.fit(train_images, train_labels,batch_size=batch_size,callbacks=[early_stopping, mcp_save],validation_split=0.1, epochs=epochs)
model.load_weights("best.mdl_wts.hdf5")
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:', test_acc)
