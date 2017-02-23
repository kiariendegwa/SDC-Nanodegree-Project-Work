#Load all mdoel training dependancies
import random
import cv2

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import math
import os
import cv2
import math

import csv
import time
import argparse
import json
#Tflow keras wrapper
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, ELU
from keras.optimizers import Adam
from keras.callbacks import Callback
import matplotlib.image as mpimg
import tensorflow as tf

#Sci-kit learn
from sklearn.model_selection import train_test_split
    
#---------------------------------------#        
#Preprocess input data
#--------------------------------------#

def preprocess_input(x):
    img_shape = (66, 200, 3)

    height = x.shape[0]
    width = x.shape[1]

    factor = img_shape[1]/float(width)

    resized_size = (int(width*factor), int(height*factor))
    x = cv2.resize(x, resized_size)
    crop_height = resized_size[1] - img_shape[0]

    return x[crop_height:, :, :]

#--------------------------------------------#
# data augmentation
#--------------------------------------------#

def random_horizontal_flip(x, y):
    flip = np.random.randint(2)
    if flip:
        x = cv2.flip(x, 1)
        y = -y
    return x, y

def random_translation(img, steering):
    trans_range = 30  # Pixel shift
    
    # Compute translation and corresponding steering angle
    tr_x = np.random.uniform(-trans_range, trans_range)
    steering = steering + (tr_x / trans_range) * 0.17

    rows = img.shape[0]
    cols = img.shape[1]
    
    #Warp image
    M = np.float32([[1,0,tr_x],[0,1,0]])
    img = cv2.warpAffine(img,M,(cols,rows))

    return img, steering


def bright_aug(img):
    # 1 Brightness augmentation
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    r_bright = .25+np.random.uniform()
    img[:,:,2] = img[:,:,2]*r_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def data_augmentation(x, y):
    # random horizontal shift
    x, y = random_translation(x, y)

    # random horizontal flip
    x, y = random_horizontal_flip(x, y)
    
    #random brightness augmentation
    x = bright_aug(x)
    
    return x, y

#-------------------------------------------------#
#Model definition and training
#-------------------------------------------------#
def covnet():
    #Return NVIDIA covnet
    
    #Parameters
    input_shape = (66, 200, 3)
    
    #Gaussian initiation
    weight_init='glorot_uniform'
    padding = 'valid'
    dropout_prob = 0.25

    # Define model
    model = Sequential()

    model.add(Lambda(lambda X: X / 255. - 0.5, input_shape=input_shape, output_shape=input_shape))

    model.add(Convolution2D(24, 5, 5,
                                    border_mode=padding,
                                    init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5,
                            border_mode=padding,
                            init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5,
                            border_mode=padding,
                            init = weight_init, subsample = (2, 2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init = weight_init, subsample = (1, 1)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,
                            border_mode=padding,
                            init = weight_init, subsample = (1, 1)))

    model.add(Flatten())
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(100, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(50, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(10, init = weight_init))
    model.add(Dropout(dropout_prob))
    model.add(ELU())

    model.add(Dense(1, init = weight_init, name = 'output'))

    # Compile it
    model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
    
    return model
#-------------------------------------------------------------------#
#Augment data set and save new csv file
def make_data_symmetrical(df):
    records = []
    for index, row in df.iterrows():
        left = row.left
        center = row.center
        right = row.right
        steering = row.steering
        
        records.append({
            'image': left,
            'steering': steering + 0.23
        })
        
        records.append({
            'image': right,
            'steering': steering - 0.23
        })
        
        records.append({
            'image': center,
            'steering': steering
        })
    
    return pd.DataFrame(data=records, columns=['image', 'steering'])

def shift_img_augmentation(df):
    df.loc[:,'random_shift'] = 0
    new_df = df[df.steering != 0].copy()
    df.loc[:,'is_shift'] = False
    new_df.loc[:,'is_shift'] = True
    
    max_shift = 30
    max_ang = 0.17
    
    def row_shift_update(row):
        random_shift = np.random.randint(-max_shift, max_shift + 1)
        row.random_shift = random_shift
        updated_steer = row.steering + (random_shift / max_shift) * max_ang
        if abs(updated_steer) > 1:
            updated_steer = -1 if (updated_steer < 0) else 1

        row.steering = updated_steer
        return row

    new_df = new_df.apply(row_shift_update, axis=1)
    out_d_f = pd.concat([df, new_df])
    #Remove all steering values beyond the bounds [-1, 1]
    out_d_f =  out_d_f[abs(out_d_f.steering)<1]
    return out_d_f
#-------------------------------------------------------------------#

#-----------------------------------------------
# Train and validation generators
#----------------------------------------------#
def train_generator(X,y, batch_size):
    while 1:
        # Declare output data
        x_out = []
        y_out = []
        
        # Get batch training data
        for i in range(0, batch_size):
            # Get random index in dataset.
            idx = np.random.randint(len(y))
            if(X[idx][:2].strip()=="C:"):
                x_i = cv2.imread(X[idx].strip())
            else:
                x_i = cv2.imread("./data/{0}".format(X[idx].strip()))

            y_i = y[idx]

            # Preprocess image
            x_i = preprocess_input(x_i)

            # Augment data
            x_i, y_i = data_augmentation(x_i, y_i)

            # Add to batch
            x_out.append(x_i)
            y_out.append(y_i)

        yield (np.array(x_out), np.array(y_out))
        
def val_generator(X, y):
    while 1:
        for i in range(len(y)):
            if(X[i][:2].strip()=="C:"):
                x_out = cv2.imread(X[i].strip())
            else:
                x_out = cv2.imread("./data/{0}".format(X[i].strip()))
            y_out = np.array([[y[i]]])
            # Crop and normalize image
            norm = lambda X: X / 255. - 0.5
            x_out = norm(x_out)
            x_out = preprocess_input(x_out)
            x_out = x_out[None, :, :, :]
            # Return the data
            yield x_out, y_out
#----------------------------------------------#
# train and save model
#----------------------------------------------#

def save_model(model):
    #Save final model in formats as specified in Udacity ruberic
    model_json = model.to_json()
    
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save('model.h5') 
        print("Saved model to disk")

def trainModel(model, train, test):
    
    #Read in data
    X_train = np.copy(train['image'])
    y_train = np.copy(train['steering'])
    
    X_test = np.copy(test['image'])
    y_test = np.copy(test['steering'])
    
    
    #Batch size 
    batch_size = 64
    
    #Numnber of epochs
    n_epochs = 1
    
    n_train_samples =len(train)
    n_val_samples = len(test)
  
    gen_train = train_generator(X_train, y_train, batch_size)
    gen_val = val_generator(X_test, y_test)

    model.fit_generator(generator = gen_train,
                        samples_per_epoch = n_train_samples,
                        validation_data = gen_val,
                        nb_val_samples = n_val_samples,
                        nb_epoch = n_epochs,
                                    verbose = 1)
    save_model(model)
    
def main():
    d_f = pd.read_csv("data/driving_log.csv") 
    #Generate new data
    img_flip = make_data_symmetrical(d_f)  
    img_shifted = shift_img_augmentation(img_flip)

    #Save new pandas df
    img_shifted.to_csv('data/balanced_driver_log.csv')
    global train
    global test
    #90% training data, 10% testing data
    train, test = train_test_split(img_shifted, test_size = 0.10)
    #load model:
    net = covnet()
    #train and save model
    trainModel(net, train, test)   
  
    
if __name__ == "__main__":
    main()