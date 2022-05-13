import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

import os


def get_network():
    num_filters = [24,32,64,128] 
    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    input_shape = (60, 41, 2)
    num_classes = 200
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size,
                padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(64, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(128, kernel_size,
                                  padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  

    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=["AUC"])
    return model


STORE_DIR = './Data/Checkpoints/'
loaded_tr = np.load(STORE_DIR+'train'+'.npz', allow_pickle=True)
loaded_val = np.load(STORE_DIR+'val'+'.npz', allow_pickle=True)
loaded_test = np.load(STORE_DIR+'test'+'.npz', allow_pickle=True)

features_tr = loaded_tr["features"]
labels_tr = loaded_tr["labels"]
features_val = loaded_val["features"]
labels_val = loaded_val["labels"]
features_test = loaded_test["features"]
labels_test = loaded_test["labels"]


x_train = np.concatenate(features_tr, axis=0) 
y_train = np.concatenate(labels_tr, axis=0) 

x_val = np.concatenate(features_val, axis=0) 
y_val= np.concatenate(labels_val, axis=0) 

model = get_network()
model.fit(x_train, y_train, validation_data = (x_val,y_val), epochs = 10, batch_size = 24, verbose = 1, shuffle=True)
model.save('./Data/Checkpoints/model')


# reconstructed_model = keras.models.load_model('./Data/Checkpoints/model')
# For loop through test data
# for loop for chunks
# model.predict
# calculate AP and append to list
# mean of that list and append to master list (AP of file)
# mean of master list = mAP

master_ap = []
for test_idx in range(len(features_test)):
    y_pred = model.predict(features_test[test_idx])
    ap_score = []
    for chunk_idx in range(len(y_pred)):
        ap_score.append(metrics.average_precision_score(labels_test[test_idx][chunk_idx],y_pred[chunk_idx]))
    master_ap.append(np.mean(ap_score))

mAP = np.mean(master_ap)
print('mAP:')
print(mAP)