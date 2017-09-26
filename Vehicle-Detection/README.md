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

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### README
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Having noticed the computational bottleneck caused by the sliding window approach. That is a situation whereby a window is iteratively passed
over each image at at time complexity of q* O(n^(m*p)), where q is the number of sliding windows,
n, the number of image pixels and (m*p) the sliding window dimensions.
Clearly this quadratic time could be shortened dramatically using a neural segmentation algorithm, wink, wink. 
The u-net design described here [U net link](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) was therefore used. This would require
a single pass through the image, with optimized GPU calculations used to carry out the inference.
The resultant net had an inference time far shorter than the HOG detector suggested by Udacity, taking 1min 23
seconds to render the video displayed in a later section. It was also a nice excuse to use a neural net.
 
![U-net architecture][image1]

The 33gb Udacity training set alongside the CrowdAi data was sufficient enough to serve as a sufficient training set without need
to apply transfer learning using a famous pre-trained covnet. Taking 1min 23s. However training took roughly 3hrs to carry out.

Samples of the training data are highlighted below, the coordinates of which are described alongsided a .csv file 
contained in the tagged data sets:

![Initial 1920 by 1200 RGB training data][image2]

This is then used to draw bounding boxes around the area of interest.

![Tagged training data][image3]

RGB images of 400 by 640 where directly feed into the neural network given the tagged images highlighted above. 
The neural net was feed training images of the unmarked driving data alongside a training set comprised of bounding boxes around vehicles.

These trained neural net resulted in the following heatmaps/normalized logits:

![Original image alongside predicted hotspots and training data][image4]

It was somewhat of a relief to find sufficient results given that no data augmentation was carried out. Theoretically this could have]
been used to increase the data set from 33GB to 100GB. This would have most certaintly resulted in better segmentation, however this is
a MVP to be re-written in Pytorch (Keras is fun for rapid protoyping).

#### 2. Explain how you settled on your final choice of HOG parameters.

The neural architecture described below was picked so that the AWS GPU memory 
would not run out given the size of the batch data being feed into it. The GPUs used I think were P2, so 16gb Memory.
The network was therefore trained on batch sizes of 1000 images(research shows that this should actually have been lower - but meh) 
and trained for 30 epochs using an ADAM optimizer, a learning rate of 1e4. Here's a pretty picture of the final architecture:

![Original image alongside predicted hotspots and training data][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A global image buffer that calculated the heatmap average over every 10 frames was used. These heatmaps where used to determine the average heatmap coordinates overwhich a bounding box was overlayed over the corresponding frames. 
This was sufficient in removing erraneous detections and crazy annoying damned jitter.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![Original image alongside predicted hotspots and training data][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The reason I chose to approach this problem using U nets is that I liked the idea of having an end to end gradient friendly algorithm. 
Also this is the 21st century damnit, if you can solve a problem by propagating gradients, you betcha it's gonna work. Also this can be extended to detecting all manner of other random objects in a training set.

