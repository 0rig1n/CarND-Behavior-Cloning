#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
First of all, It's very important to make the model concentrate, the trees mountains in the image make less contribute on decision making and can easyly distract the model. so I cropped the image before the model so that the model can concentrate. 

I tried Nvidia architecture at the first time, since this model may work well on these tasks. But most importantly I have a personal project that I was forced to implement Nvidia architecture for a similar task, I can do 2 project at the same time. 

My model mostly based on Nvidia architecture, The network consists of 9 layers, including 5 convolutional layers and 4 fully connected layers.

The first three layer are similar, I use strided convolutions in the first three convolutional layers with a 2 × 2 stride and a 5 × 5 kernel, they were designed to perform feature extraction, a good benifit of use a well-known architecture is the model will be fast small and effcient;

Also the model may need some overall viewpoint for autonomous drive task, that's what the convolutional layer capable for, so the model apply two more non-strided convolutional layer with a 3 × 3 kernel size.

Finally goes 4 fully-connected layed , they will make decesion based on the feature map that  convolutional extract, I tried initialize the weight so the model may learn with a good start point.

The model includes RELU layers to introduce nonlinearity (code line 109 - 113), and the data is normalized in the model using a Keras lambda layer (code line 105-107). 

####2. Attempts to reduce overfitting in the model

I didn't tried dropout in my Conv layer, because it has little unit and the randomness may cause training unstable.Also, I'm not very afraid of overfitting because the model has less unit on each layer, and the model is not very deep, this may avoid overfitting with a smaller Epoch.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29-30). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning(model.py line 81-88)

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124).
The weight initiate with mean=0 and std=0.1, according that after normalize the data, the data have 0 mean value and low standard deviation.
BATCH_SIZE set to 128 so that the model can get a better result and learn faster.
CORRECTION=0.25 seems to get a better result.
There is a hyperparamters called SCALE, that's my adjustment for my history dead model, I'will discuss it at *Solution Design Approach* part.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, I make a mistake at recovering part the first time, I flip the image of left and right sides camera, but forget to exchange these two, so my model drive badly at beginning, but I fix it and the model turns OK

There is a long story for me and my training data, I will talk about it in next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

At very beginning I concentrate on data collection, I've collect about 2G data at first try, the model drives OK but don't kown what to do on bend, so I collect 6G data, train this network takes me about a week, and model drive ridiculous, it sucessfully passed the two most difficult sharp turn, drive smoothly on straight line but failed on first turn, which is less sharp and easy to drive. This makes me frustrated and times runing out.

After discuss with my friend,I believe that the sample data is enough for the task, I should be more focus on the model instead of data, this helps me concentrate. 

Then it was not far from my final model, My first step was to use a convolution neural network model similar to the Nvidia Architecture, I thought this model might be appropriate because this model was used in a real autonomous drive, it must be capable for this task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and also an even lower mean squared error on the validation set. This implied that the model may perform good. So I tried my model on the simulator.

In simulator, the model pass the first turn, then get confused on the bridge and turns little on these two most difficult sharp turn. Then I get an idea: since the model was confused on these difficulty and make desision with great hesitation, maybe I can SCALE the label data a little bit, so my model may get more confidence. Although this may cause a little bit fluctuation on straight line, but that's acceptable.

So I add a new hyperparameter called SCALE, to give my model more confidence. I then tuning the hyperparameter, retrain the model and test it on simulator, wow this time the model drives much better and meet the specification! 

This became my final model, I waste too much time on collecting data in this project and I have to submit it in time. But I will try more attempt after this project, it's very interesting and practical.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

------------------------Model summary---------------------------
|Layer (type)               |Output Shape           |Param #   |
|---------------------------|-----------------------|----------|
|cropping2d_1 (Cropping2D)  |(None, 65, 320, 3)     |   0      |
|lambda_1 (Lambda)          |(None, 65, 320, 3)     |   0      |
|conv2d_1 (Conv2D)          |(None, 31, 158, 24)    |  1824    |
|conv2d_2 (Conv2D)          |(None, 14, 77, 36)     | 21636    |
|conv2d_3 (Conv2D)          |(None, 5, 37, 48)      | 43248    |
|conv2d_4 (Conv2D)          |(None, 3, 35, 64)      | 27712    |
|conv2d_5 (Conv2D)          |(None, 1, 33, 64)      | 36928    |
|flatten_1 (Flatten)        |(None, 2112)           |    0     |
|dense_1 (Dense)            |(None, 100)            | 211300   |
|dense_2 (Dense)            |(None, 50)             |  5050    |
|dense_3 (Dense)            |(None, 10)             |   510    |
|dense_4 (Dense)            |(None, 1)              |   11     | 
|Total params:| 348,219     |
|Trainable params:| 348,219 |
|Non-trainable params:| 0   |

_________________________________________________________________

####3. Creation of the Training Set & Training Process

first,I randomly shuffled the data set and put 20% of the data into a validation set.
Then I apply a recovery drive, I used left and right images as center image with a correction paramter, this may help model to stay at the center of the road.
In order to augment the data sat, I also flipped images and angles thinking that this would create more data, but I made a funny mistake here: the images flipped from left and right cameras should exchange, but I didn't at the first time, and the model collapse.

I sample training data for training the model. Because I believe that the sample data is enough for the task. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z according to when the training loss hardly decre ase, I used an adam optimizer so that manually training the learning rate wasn't necessary.
