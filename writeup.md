# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/image1.jpg "Center lane driving"
[image2]: ./output_images/image2.jpg "Recovery Image"
[image3]: ./output_images/image3.jpg "Recovery Image"
[image4]: ./output_images/image4.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* training.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an implementation of the Nvidia neural net architecture. The data is first normalized in the model using a Keras lambda layer and then the image is cropped to remove some of the upper and lower regions of the image which are not relevant to driving, using a Keras cropping layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated data sets with a flipped image to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. However I settled on the number of training epochs based on the training and validation loss plots.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and some driving in the reverse.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a somewhat proven model to start with and apply small refinements for this particular problem.

My first step was to use a convolution neural network model similar to the Nvidia architecture. I thought this model might be appropriate because it has been well researched and emerged as a good choice when applied to self-driving cars.

Since I started with a fairly complex model, I ended up facing quite a few OOM(out of memory) exceptions on system. To combat that I tried a simpler Lenet based architecture, but it was performing poorly in the driving simulations. So I switched back to the Nvidia architecture and discovered that applying subsampling to the convolution layers would reduce the number of tunable params and hence reduce the memory load. Indeed, after applying the subsample parameter, I was able to train the model without a hitch.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or would get stuck near the bridge. To improve the driving behavior in these cases, I augmented the training data with more examples of recovery in those specific spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
- A cropping layer to remove extraneous parts of the image
- 5 2D convolution layers, with 24 to 64 output dimensions, a kernel size of 5x5 and each followed by an RELU activation layer. Some of the layers are subsampled to conserve memory.
- These are followed by a flatten layer.
- Finally, 4 Dense layers with 100, 50, 10, and 1 outputs lead to the model output.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to return to the center lane when it swerves too far to the edge. I added a few more after testing the initial performance of the model and observing what parts of the track were more difficult. These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]

After the collection process, I had about 3500 data points, which I then augmented with left and right camera images and also the flipped images of the center camera, totaling to a about 14000 training images.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 which I settled upon based on observations from the the training and validation loss v/s number of epochs plot. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
