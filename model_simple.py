import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

import os

OVERLAP_TO_LOAD = 50

def get_network():
    num_filters = [24,32,64,128] 
    pool_size = (2, 2) 
    kernel_size = (3, 3)  
    input_shape = (60, 41, 2)
    num_classes = 200
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(24, kernel_size, padding="same", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

    model.add(keras.layers.Conv2D(32, kernel_size, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(64, kernel_size, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))  
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(keras.layers.Conv2D(128, kernel_size, padding="same"))
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
loaded_tr = np.load(STORE_DIR+'train_'+str(OVERLAP_TO_LOAD)+'.npz', allow_pickle=True)
print('='*20 + 'Train npz loaded!' + '='*20)
loaded_val = np.load(STORE_DIR+'val_'+str(OVERLAP_TO_LOAD)+'.npz', allow_pickle=True)
print('='*20 + 'Valid npz loaded!' + '='*20)
loaded_test = np.load(STORE_DIR+'test_'+str(OVERLAP_TO_LOAD)+'.npz', allow_pickle=True)
print('='*20 + 'Test npz loaded!' + '='*20)

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
model.fit(x_train, y_train, validation_data = (x_val,y_val), epochs = 10, batch_size = 24, verbose = 2, shuffle=True)

model.save('./Data/Checkpoints/model/model_simple')

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