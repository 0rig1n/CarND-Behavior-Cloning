
# coding: utf-8

# In[5]:
# Import all module that we need, also initiate the hyperparamter of the model  
import os
import pandas as pd
import cv2
_DATA_DIR_ = "./data"
TEST=True
###  Hyperparameter

BATCH_SIZE=32 
CORRECTION=0.25
EPOCH=3
LEARNING_RATE=0.001
MEAN=0
STD=0.1
if(TEST):
    SEED=3
else:
    SEED=None


# In[6]:
# Read the data_log

data_log=pd.read_csv(_DATA_DIR_+"/driving_log.csv")
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data_log, test_size=0.2)
print("Step1 Complete: Load data Sucessfully!")


# In[7]:
# Preprocess the data, load the image into memory, get ready for training
# Technic include : data argumentation ; use multiple camera data for training
# other technic are used in model below

import numpy as np
import sklearn
from sklearn.utils import shuffle
def preprocess(samples):
    num_samples = len(samples)
    correction = CORRECTION # this is a parameter to tune
    shuffle(samples)
    images = []
    angles = []
    for batch_sample in samples.values:
        center_name = './data/IMG/'+batch_sample[0][4:]
        left_name = './data/IMG/'+batch_sample[1][4:]
        right_name = './data/IMG/'+batch_sample[2][4:]
                
        center_image = cv2.imread(center_name)
        left_image = cv2.imread(left_name)
        right_image = cv2.imread(right_name)
                
        center_angle = float(batch_sample[3])
        left_angle = center_angle + correction
        right_angle = center_angle - correction
                
        center_image_flipped = np.fliplr(center_image)
        center_measurement_flipped = -center_angle
        left_image_flipped = np.fliplr(right_image)
        left_measurement_flipped = -right_angle
        right_image_flipped = np.fliplr(left_image)
        right_measurement_flipped = -left_angle
                
        images.extend([center_image,left_image,right_image,center_image_flipped,left_image_flipped,right_image_flipped])
        angles.extend([center_angle,left_angle,right_angle,center_measurement_flipped,left_measurement_flipped,right_measurement_flipped])

    X_train = np.array(images)
    y_train = np.array(angles)
    return (X_train,y_train)
X_train,y_train=preprocess(train_samples)
X_validation,y_validation=preprocess(validation_samples)
print("Step2 Complete: Sucessfully define the geneator!")


# In[8]:

### Tuning Hyperparameter
BATCH_SIZE= 128
CORRECTION=0.25
EPOCH=5
LEARNING_RATE=0.001
SCALE=1.5
MEAN=0
STD=0.1


# In[9]:
# Build the model, Nvidia arch, because it's small, fast and effcient.
# Use: cropping, data normalization  
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras import optimizers
from keras import initializers

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(65, 320, 3),
        output_shape=(65, 320, 3)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
init1=initializers.TruncatedNormal(mean=MEAN, stddev=STD, seed=SEED)
model.add(Dense(100,kernel_initializer=init1))
init2=initializers.TruncatedNormal(mean=MEAN, stddev=STD, seed=SEED)
model.add(Dense(50,kernel_initializer=init2))
init3=initializers.TruncatedNormal(mean=MEAN, stddev=STD, seed=SEED)
model.add(Dense(10,kernel_initializer=init3))
init4=initializers.TruncatedNormal(mean=MEAN, stddev=STD, seed=SEED)
model.add(Dense(1,kernel_initializer=init4))

adam=optimizers.Adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam)
model.summary()
print("Step3 Complete: Sucessfully build the model!")
# ------------------------Model summary---------------------------
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
#_________________________________________________________________
#lambda_1 (Lambda)            (None, 65, 320, 3)        0         
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
#_________________________________________________________________
#conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
#_________________________________________________________________
#conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
#_________________________________________________________________
#conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
#_________________________________________________________________
#conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
#_________________________________________________________________
#flatten_1 (Flatten)          (None, 2112)              0         
#_________________________________________________________________
#dense_1 (Dense)              (None, 100)               211300    
#_________________________________________________________________
#dense_2 (Dense)              (None, 50)                5050      
#_________________________________________________________________
#dense_3 (Dense)              (None, 10)                510       
#_________________________________________________________________
#dense_4 (Dense)              (None, 1)                 11        
#=================================================================
#Total params: 348,219
#Trainable params: 348,219
#Non-trainable params: 0
#_________________________________________________________________

# In[10]:
# Train the network 
print("start training")
model.fit(X_train,y_train*SCALE,batch_size=BATCH_SIZE,epochs=EPOCH,validation_data=(X_validation,y_validation*SCALE))


# In[6]:




# In[11]:
# Save the model
model.save("model.h5")

