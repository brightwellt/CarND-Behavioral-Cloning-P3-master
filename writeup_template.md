#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ModelVisualisation.png "Model Visualization"
[image2]: ./examples/GrayScaling.png "Grayscaling"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/NotFlipped.jpg "Normal Image"
[image7]: ./examples/Flipped.jpg "Flipped Image"
[image8]: ./examples/Left.jpg "Left Image"
[image9]: ./examples/Center.jpg "Center Image"
[image10]: ./examples/Right.jpg "Right Image"
[image11]: ./examples/Losses.PNG "Losses"
[image12]: ./examples/CenterLaneDriving.png "Center Lane Driving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Project3-Submitted.ipynb (the Jupyter notebook that the model was originally written in)
* drive.py for driving the car in autonomous mode (modified from the original)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 - a video of a complete autonomous lap on track 1 

It can be found here
[Link to GitHub](https://github.com/brightwellt/CarND-Behavioral-Cloning-P3-master)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The model code was developed in a jupyter notebook running on an AWS instance. The notebook is attached.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based upon an architecture published by Nvidia here
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
and referenced in the course notes.
It consists of a convolution neural network with 3x3 / 5x5 filter sizes and depths between 32 and 128 (model.py lines 58-73) 

The model includes RELU layers to introduce nonlinearity (code line 62 onwards) and the data is normalized in the model using a Keras lambda layer (code line 59). The normalized images are cropped to remove the sky. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 61, 65, 68). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving in both directions around the track. The mouse was used for steering.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple 1 layer architecture, progress to the NVidia architecture, and from there focus on getting a good, properly pre-processed dataset.

I thought the NVidia example model would be a good starting point. It is appropriate because it is specifically designed for producing steering commands for a self-driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there were some dropout layers.

Then I reran the model, reviewing the training / validation errors and also the number of epochs. 

The final step of the process was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, or seemed less stable. One on occasion, the car even drove around the off-road area towards the end of the track! 

To improve the driving behavior in these cases, I recorded additional data; driving around the track in the opposite direction, and also adding specific extra sections for the trouble-spots. Starting from a position towards the edge of the track and driving back towards the middle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...
Lambda layer (normalisation of data)
Cropping2D (remove the sky)
Dropout (0.2)
Convolution2D (24, 5x5, relu)
Convolution2D (36, 5x5, relu)
Convolution2D (48, 5x5, relu)
Dropout (0.2)
Convolution2D (64, 3x3, relu)
Convolution2D (64, 3x3, relu)
Dropout (0.2)
Flatten()
Dense(100)
Dense(50)
Dense(10)
Dense(1)

Here is a visualization of the architecture without the dropout / cropping layers (note: visualizing the architecture is optional according to the project rubric)

![NVidia][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center driving][image12]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover itself when it started to veer off the track in Autonomous Mode. These images show what a recovery looks like starting from ... :

![On side of track][image3]
![Turning back in][image4]
![Back to middle][image5]

I then added in the left and right camera images that the simulator records. This gave me extra data representative of a car veering across the track. For each image, I corrected the steering angle; adding a correction factor to left images and subtracting this factor for right images. These images show left, right and center images:

![Left][image8]
![Center][image9]
![Right][image10]

During my investigations, I repeated this process on track two in order to get more data points. This also produces a more robust model, able to handle a wider variety of circumstances. However I also found that the track two data was not necessary for a successful drive around track one; especially when grayscaling the data (see below).


To augment the data set, I also flipped images and angles thinking that this would double the amount of images... For example, here is an image that has then been flipped:

![Normal image][image6]
![Flipped image][image7]

Etc ....

After the collection process, I had 18,996 number of data points. I then preprocessed this data by grayscaling the data. 
![GrayScaled Image][image2]
This meant altering the supplied drive.py to grayscale the input data from the simulator so that it would match the model's inputs. I did look at using a Keras ImageDataGenerator to build the grayscale operation into the model itself, and will investigate further outside the scope of this writeup.
Grayscaling the data significantly improved the model performance; reducing the error further and ensuring that it could reliably steer a car around track one. It also reduced the effect of altering my left / right correction factor, thus implying that it increased robustness. 
The video, "run1.mp4" illustrates the saved model.h5 driving around track one..

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 as evidenced by plotting the mean squared error losses during training and validation. Beyond 9 epochs the training loss flattened out. 
![Losses][image11]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
