##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/u-net-architecture.png
[image2]: .report_images/training_data.png
[image3]: ./report_images/more_training_data2.png
[image4]: ./report_images/Predictions.png
[image5]: ./report_images/Network_Architecture.png
[image6]: ./report_images/last_frame.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Having noticed the computational bottleneck caused by the sliding window approach, requiring the window to check multiple image scales. I decided to instead implement a neural segmentation algorithm based on the u-net design described here [insert link here]. 

![U-net architecture][report_images/u-net-architecture.png]

Initially it was thought that this could be used to preliminary pick hot spots within an image which would significantly reduce the aforementioned computational bottleneck.

However after tweaking the U net image segmentation net, it was found to be sufficient to the task at hand. The 33gb Udacity training set alongside the CrowdAi data was sufficient enough to categorize video data. Taking 1min 23s. However training took roughly 3hrs to carry out.

The training data is highlighted below, the coordinates of which are described alongsided a .csv file contained in the tagged data set:

![Initial 1920 by 1200 RGB training data][report_images/training_data.png]

This is then used to draw bounding boxes around the area of interest.

![Tagged training data][report_images/more_training_data2.png]

RGB images of 400 by 640 where directly feed into the neural network given the tagged images highlighted above. The neural net was feed training images of the unmarked driving data alongside a training set comprised of bounding boxes around vehicles.

These post training resulted in the following heatmaps:

![Original image alongside predicted hotspots and training data][report_images/Predictions.png]

Given that the original goal was to simply use this method as a means to get rid of the computational sliding window bottle neck
color spaces were not explored given that the algorithm worked straigh out of the box.

####2. Explain how you settled on your final choice of HOG parameters.

The neural architecture described below was picked so that the AWS GPU memory would not run out given the size of the batch data being feed into it. The network was trained on batch sizes of 1000 images and trained for 30 epochs using an ADAM optimizer with a learning rate of 1e4 and training and test generators. The final architecture is described below:

![Original image alongside predicted hotspots and training data][report_images/Network_Architecture.png]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A global image buffer that calculated the heatmap average over every 10 frames was used. These heatmaps where used to determine the average heatmap coordinates overwhich a bounding box was overlayed over the corresponding frames. This was used to sufficiently remove erraneous detections.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![Original image alongside predicted hotspots and training data][report_images/last_frame.png]

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The reason I chose to approach this problem using U nets is that I liked the idea of having an end to end system that could take in a variety of data that a SDC could use not limited only to the detection of vehicles - I have also worked on SIFT detectors similar to HOG detectors and wanted to impelement a newer algorithm. 

The Udacity training set also had street lights highlighted in each frame alongside pedestrains. This made it more interesting to experiment with in its entirety. Admittedly the U-net had draw backs with regards to training times and a lack of adequately tagged training set containing primarily street lights or pedestrians. However I see the U-net as an algorithm used to highlight regions of interest within an image fairly first. These regions of interest could then be categorized by other algorithms as trucks, cars, pedestrians etc.

