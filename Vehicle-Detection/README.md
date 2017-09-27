**Vehicle Detection Project**

The goals stated goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/u-net-architecture.png
[image2]: ./report_images/training_data.png
[image3]: ./report_images/more_training_data2.png
[image4]: ./report_images/Predictions.png
[image5]: ./report_images/Network_Architecture.png
[image6]: ./report_images/last_frame.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### README
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Having noticed the computational bottleneck caused by the sliding window approach. That is a situation whereby a window is iteratively passed
over each image at at time complexity of O(xy(n^(n)), where n is the number of sliding windows and x and y, the width and height of the image.
Clearly this polynomial time could be shortened dramatically using a neural segmentation algorithm. 
The u-net design described here [U net link](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) was therefore used. This would require
a single pass through the image, with optimized GPU calculations used to carry out the inference.
The resultant net had an inference time far shorter than the HOG detector suggested by Udacity, taking 1min 23
seconds to render the video displayed in a later section. It was also a nice excuse to use a neural net.
 
![U-net architecture][image1]

The 33gb Udacity training set alongside the CrowdAi data was sufficient enough to serve as a training set without need
to apply transfer learning using some random famous pre-trained covnet for one arm of the u-net or further data augmentation. 
Training took roughly 3hrs to carry out given the above data sets.

Samples of the training data are highlighted below, the coordinates of which are described alongsided a .csv file 
accompanying the image data:

![Initial 1920 by 1200 RGB training data][image2]

This is then used to draw bounding boxes around the area of interest.

![Tagged training data][image3]

Resized RGB images of 400 by 640 where directly feed into the neural network given the tagged binary masks(The final training masks fed into the network where 1*400*600 - should we have had multiple classes of objects
say, 3, the training mask would have been 3*400*600 - you get the drift).


The trained neural net resulted in the following heatmaps/logits after training:

![Original image alongside predicted hotspots and training data][image4]

It was somewhat of a relief to find  the sufficient results given that no data augmentation was carried out. Theoretically this could have]
been used to increase the data set from 33GB to 100GB - by adding shadows, random rotations etc. 
This would have most certaintly resulted in better segmentation, however this is
a MVP to be re-written in Pytorch (Keras and Jupyter are fun for rapid protoyping).

#### 2. Explain how you settled on your final choice of HOG parameters.

The neural architecture described below was picked as we are dealing with image segmentation, and it is a well known tried and tested architecture. 
The network was trained on batch sizes of 1000 images(research shows that this should actually have been lower - but meh) 
and trained for 30 epochs using an ADAM optimizer, with a learning rate of 1e4. Here's a pretty picture of the final architecture:

![Original image alongside predicted hotspots and training data][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4). Inference time took 1 min 23 seconds.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A global image buffer that calculated the heatmap average over every 10 frames was used. These heatmaps where used to determine the average heatmap coordinates overwhich a bounding box was overlayed over the corresponding frames. 
This was sufficient in removing erraneous detections and crazy annoying damned jitter.

Here's an example result showing the heatmap from a series of frames of the video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![Original image alongside predicted hotspots and training data][image6]

---

### Discussions

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The reason I chose to approach this problem using U nets is that I liked the idea of having an end to end algorithm with fast inference times. 

