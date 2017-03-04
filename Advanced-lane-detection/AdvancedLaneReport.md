**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Original_image.jpg "Original"
[image2]: ./output_images/Undistorted_Original_image.jpg "Undistorted"
[image3]: ./output_images/test_image.jpg "Test image"
[image4]: ./output_images/undistorted_test_image.jpg "Undistorted test image"
[image5]: ./output_images/birds_eye_images.png "Birds eye view"
[image6]: ./output_images/yellow_and_white_filter_image.png "Color filter"
[image7]: ./output_images/color_sobel_filter_images.png "Combined sobel and color filter"
[image8]: ./output_images/polynomial_fit.png "Polynomial fit"
[image9]: ./output_images/lane_overlay.png "Lane curve and lane overlay"
[image10]: ./output_images/perspective_full_figure.png "Perspective check"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/kiariendegwa/SDC_Nano_degree/blob/master/Advanced-lane-detection/AdvancedLaneReport.md)

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "advanced-Lane-detection.ipynb" (The first cell of the notebook).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Given the original image
![alt text][image1]
Given the undistored image
![alt text][image2]
###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The functioning code is contained within the notebook, "Advanced-lane-detection.ipynb"
The process involved converting the image to HSV and then isolating the yellow and white lanes from the image (using the functions, "color_mask" and "cv2.bitwise_or").
These filters where then combined and applied to an input HSV image, the output was consequently turned into the gray image as displayed below. 

Sobel filters (using the "abs_sobel_thresh" function) were also applied onto the image, trial and error was used to determine the adequate rations. The eventual filter was combined with the yellow and white lane filters described in brief above. 

Again the code base for these transformations is contained within the ipython notebook labelled "Advanced-lane-detection-ipynb". The cells performing these color filter and sobel transformations are contained within the cells 12 and 8. You can't miss them as they're appropriated labelled within the notebook.

Here's an example of my yellow and white color filters applied to a birds eye perspective image.
#Original birds eye view image
![alt text][image5]
#Yellow and white color filter applied to image
![alt text][image6]
#Sobel filters then added onto the yellow and white filters
![alt text][image7]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp_birds_eye_view`, which appears in lines 1 through 8 in the file `Advanced-lane-detection.ipynb` (This is contained within cell 11).  The `corners_unwarp_birds_eye_view` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points generated by the image undistortion function detailed above.  I chose the hardcode the source and destination points in the following manner:

src = np.float32(
    [[120, 720],
     [550, 470],
     [700, 470],
     [1160, 720]])

dst = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]])


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Again these steps are contained within the notebook, `Advanced-lane-detection.ipynb` in the cell 22 and the function labelled `video_pipeline` 

The lane detection was carried out by using histogram analysis on the resulting image post processing: i.e. having passed the image through the appropriate steps listed below:
   
*image undistortion, 
*color and sobel filters,
*image warping - birds eye view

The image is split into 2 images along its x axis, splitting the image into a left and right lanes. Each of these is then split into 9 windows, each of which has its histrogram plotted i.e. occurence of pixels larger than 0.

These resulting histograms are used to initially find the center points of the input warped gray image by performing a blind search. These resulting center points are then used to calculate the resultant 2nd order polynomial using the `np.polyfit` function whose values are then used to calculate the polygon used to highlight road lanes. 


![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this again within the `Advanced-lane-detection.ipynb` the function used to calculate the curvatures of the left and right lanes are contained within the function `get_curvature`. This is based of the calculas derivation described in 
[Derivation of centre of curvature](http://www3.ul.ie/~rynnet/swconics/E-COC.htm)


Whereas the location of the camera was calculated by finding the centre point between the polynomial curves of the left and right lanes. This is carried out within the function `video_pipeline`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This was carried out again in the notebook `Advanced-lane-detection.ipynb` in the function labelled `video_pipeline`.  Here is an example of my result on a test image:

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline implemented above is a MVP given the time constraint of the project duration. I feel that further techniques could be used to further improve results. 

The sobel filters could have been made more robust with more fine-tuning.

Secondly Hough transforms could have been used to further improve the robustness of the sliding windows and histogram method described above.

Thirdly, more robust light filtering could have been used by exploring the YUV color space for effective lane extraction, this combined with the HLS and HSV color space could have resulted in more robust filters.

Have passed this pipeline through the challenge video I can't help but think the main challenge given such a problem is based purely in finetuning robust color filters possibly with a more user friendly interface to tweak all the values within filters.
