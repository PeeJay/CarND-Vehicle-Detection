
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/notcar.png
[image3]: ./examples/hogvis.png
[image4]: ./examples/hog-sub.jpg
[image5]: ./examples/pipeline.png
[image6]: ./examples/1.png
[image7]: ./examples/2.png
[image8]: ./examples/3.png
[image9]: ./examples/4.png
[image10]: ./examples/5.png
[image11]: ./examples/6.png
[image]: ./examples/.png
[image]: ./examples/.png
[video1]: ./project_video.mp4


## Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook [Here](https://github.com/PeeJay/CarND-Vehicle-Detection/blob/master/Code.ipynb)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![image1]![image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, but found them to be extremly sensitive to the point where I couldn't get detection working correctly, so I left them at the settings given in the code samples.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is contained in cell 5. To train the SVM I first extract the HOG features for car and non-car images using the functions in cell 3, and stack them into a vector suitable for training. These vectors are then split 90/10 into a training/test set, and scaled. The LinearSVC function from sklearn is then used to train a linear classifier.

I also added functionality save/load the classifier from a pickle file to save time while experimenting.

## Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My code implements a HOG sub-sampling window search. This works by dividing the search region into small windows (called blocks). The HOG features are computed for each block. These HOG feature blocks are sub-sampled into smaller sections (called cells), which can then be recombined into many different blocks by using ajacent cells. This avoids computing the HOG features multiple times when using overlaping windows.

The diagram below illustrates the technique, with a recreated block shown in yellow.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. By experimentation I settled on using scales of 1, 1.5, 2 and 2.6.

 I tried experimenting with the HOG parameters and sub-sampling window sizes, but everything I did just made it worse so I used the default values given in the code samples. Even then, I only just managed to get it (mostly) working.

 I also tried multi-threading the window search function in the hope of going from 8s to 2s per frame by searching all 4 scales simultaneously, but it didn't speed up anything.


![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I had is that the whole pipeline and all the parameters are so incredibly sensitive to change, so I had a lot of trouble getting enough positive detections with minimal false detections. I can only imagine that the pipleline would fail quite badly if used on a different video.

One of the problems is that the classifier will not work very well on vehicles it was not trained on, particulary if they are significantly different in shape.

If I had to do this again I would take the deep learning aproach, as from my experience it seems able to cope far better with imprefect input and training data.

