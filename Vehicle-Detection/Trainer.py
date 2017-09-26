#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle 
import time
import pandas as pd


#checkout pandas dataframe labels
pd_dframe = pd.read_csv('object-detection-crowdai/labels.csv')
print('Size of image data set ', pd_dframe.size)
pd_dframe.head(n=10)

#We don't need pedestrians in data set so drop all rows with the pedestrian label, also drop Preview URL
pd_dframe = pd_dframe[pd_dframe.Label != 'Pedestrian']
del pd_dframe['Preview URL']

#Add folder to point to correct training folder
pd_dframe['File_Path'] =  'object-detection-crowdai/' + pd_dframe['Frame']

#There is something off about the coordinates, particularly 
#with regards the min max values of bounding box coordinates, 
#further investigations shows that they should be switched arounds as follows:
pd_dframe.columns = ['ymin', 'xmin', 'ymax', 'xmax', 'Frame', 'Label', 'File_Path']
pd_dframe.head()

pd_dframe2 = pd.read_csv('object-dataset/labels.csv', delimiter=r"\s+",  
                        names= ['Frame',  'xmin', 'xmax', 'ymin','ymax', 'ind', 'Label','RM'])
del pd_dframe2['RM']
del pd_dframe2['ind']
pd_dframe2 =  pd_dframe2[pd_dframe2.Label != 'pedestrian']
#Correct coordinates before concatenating
pd_dframe2.columns = ['Frame', 'ymin', 'xmin', 'ymax', 'xmax', 'Label']
pd_dframe2['File_Path'] =  'object-dataset/' + pd_dframe2['Frame']

#Concatenate both dataframes into 1:
pd_dframe = pd.concat([pd_dframe2,pd_dframe]).reset_index()
del pd_dframe['index']


import matplotlib.patches as patches
def get_image_name(df,ind):
    #Get image and resize then return all bounding box coordinates
    
    #Image size
    size=(640,400) 
    #size =(1920, 1200)
    file_name = df.iloc[ind]['File_Path']
    img = cv2.imread(file_name)
    img_size = np.shape(img)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    #Resize image
    img = cv2.resize(img,size)
    img_size_post = np.shape(img)
    
    #Load frame name
    name_str = df.iloc[ind]['Frame']
     
    #Get all box coordinates of given image
    bb_boxes = df[df['Frame'] == name_str].reset_index()
     
    #Reshape boxes to fit new image dim
    bb_boxes['xmin'] = np.round(bb_boxes['xmin']/img_size[1]*img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax']/img_size[1]*img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin']/img_size[0]*img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax']/img_size[0]*img_size_post[0])
    
    return img, bb_boxes
 
def get_mask_seg(img,bb_boxes_f): 
    # Get bounding masks   
    img_mask = np.zeros_like(img[:,:,0])
    
    for i in range(len(bb_boxes_f)):
        bb_box_i = [bb_boxes_f.iloc[i]['ymin'],bb_boxes_f.iloc[i]['xmin'],
                bb_boxes_f.iloc[i]['ymax'],bb_boxes_f.iloc[i]['xmax']]
        img_mask[bb_box_i[1]:bb_box_i[3],bb_box_i[0]:bb_box_i[2]]= 1.
        img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))
    return img_mask

def plot_im_bbox(im,bb_boxes):
    #Draw bounding boxes around vehicles
    f, axarr = plt.subplots(1, 2)
    axarr[1].imshow(im)
    axarr[0].set_title('Original image')
    axarr[0].imshow(im)
    for index, row in bb_boxes.iterrows():
        rect = patches.Rectangle(
        (row['ymin'], row['xmin']),   # (x,y)
        row['ymax']-row['ymin'],      # width
        row['xmax'] - row['xmin'],    # height
        )
        axarr[1].add_patch(rect)
        axarr[1].set_title('Cars with bounding boxes')
    plt.show()

img_rows = 400
img_cols = 640
    
def generate_train_batch(data, batch_size = 32):
    #Where data is a pandas dataframe
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            img,bb_boxes = get_image_name(data, i_line)
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
            
        yield batch_images, batch_masks
        
def generate_test_batch(data,batch_size = 32): 
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            img,bb_boxes = get_image_name(data, i_line)
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
            
        yield batch_images, batch_masks


#Import all scipy, keras and tflow modules
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label
import time


#next up we need to define the (intersection over union) IOU as used when training images segmentation:
smooth = 1.
def IOU_calc_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)    
    return -2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def IOU_calc(y_true, y_pred):
    return -IOU_calc_loss(y_true, y_pred)

### Defining a small Unet
def get_small_unet():
    inputs = Input((img_rows, img_cols,3))
    #Normalize images
    
    #Normal convolution downsampling
    inputs_norm = Lambda(lambda x: x/127.5 - 1.)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

    #Up convolve and merge previous downsampling
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv9)

    #Add sigmoid layer to get segmentaion probability
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model

training_gen = generate_train_batch(pd_dframe,1)
smooth = 1.
model = get_small_unet()
model.compile(optimizer=Adam(lr=1e-4),
              loss=IOU_calc_loss, metrics=[IOU_calc])
model.summary()

history = model.fit_generator(training_gen,
            samples_per_epoch=1000,
                              nb_epoch=50, verbose=1)

model.save('Vehicle_detect_SmallUnet_two.h5')