##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./writeup_images/car.png
[image2]: ./writeup_images/non_car.png
[image3]: ./writeup_images/hog_example.png
[image4]: ./writeup_images/sliding_windows.png
[image5]: ./writeup_images/car_patches_and_heatmap.png
[image6]: ./writeup_images/heatmap_example.png
[video1]: ./submission.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The features are extracted in lines 82-96 of generate_features.py. The function in which all the features are extracted 
is pretty straight-forward. The 3 types of features used are in order color-binning features (used 32x32), histogram 
features (used 32 bins between 0 and 256 used on LUV and HLS), and HOG features, which are described in more detail below.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt_text][image2]

The dimensions of the hog features were 8 orientations, 8 pixels per cell and 2 cells per block. On a 64x64 grid, 
this leads to 7x7x2x2x8 dimension hog features. Ultimately, I did not do much experimentation on these parameters
because an initial training session gave me above 99% accuracy on the validation set. The choice of color space,
which I noted above, required more experimentation. 

This is one of the areas with the greatest potential for improvement. The fastest submission will use a small number
of features and a small number of search windows. I believe that finding the minimal size of the HOG features is 
especially useful in this regards.


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, 
and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what 
the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I ended up using LUV as the primary color space. It does quite well in distinguishing the cars from the roads. 
These are binned and histograms and hog features are generated. I also use some HLS features and created histograms 
using this color space, though I didn't create hog features for this.
On the other hand, I found no evidence that HLS improved performance noticeably. 

The first iteration of the project I used in total 3 sets of HOG features. The first was completed converted RGB converted
to grayscale. The other two were the h and s of HLS. On a training set in which I was mindful to group similar images 
together, this produced 99.7% validation accuracy. However, when I used this model on the video, it had a tendency to 
label the road and shaded areas as cars. This led me to try the YUV color space which I found performed far better. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A linear SVM was trained using function on line 171 in train_model.py. I used a parameter of c=1 and the default measure 
(since I found that the classifier was already quite accurate.) Some playing around with the optimization suggested, 
however, that c=10 gave the best accuracy. This, of course, suggests that the training data is pretty good and the 
model does not overfit on it (because c=10 corresponds to less regularization). I also normalized all of the input 
(dividing by the standard deviation and centering it around zero) because SVMs are sensitive to scale.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function draw_all_detected_vehicles2 starting on line 154 in generate_windows.py is where the sliding window search
is implemented. The bulk of the work done by this function is done by find_centroids on line 88. This function in turn
leans on generate_feature_indices on line 11. 

generate_feature_indices generates the features in the 'batch-sliding-window HOG' search. It returns two pairs of
values. The first pair gives the indices of a group of pixels in the HOG window. The second pair gives the indices of 
the corresponding window in the original image. 

find_centroids handles the pipeline, converting the input image into the right format so that it can be fed into the 
classifier. The coordinates of the windows that are identified as vehicles are then filtered and collected. 


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For the final video, I searched 8 scales. These scales are defined starting in line 171 in generate_windows.py. These 
scales were inserted after a good deal of experimentation. This was to guarantee that there were enough activations 
in various regions of the image to get enough activity in the heat map for a bounding box to appear. Unfortunately,
having all of these regions slowed the process considerably. It takes 3 seconds per iteration to send a 
frame through the pipeline. 

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The primary filter uses the heat map to define a bounding box (bbox) for a car. If there are at least two boxes
overalapping, the pixel is kept. Then in a contiguous region, the 2-d minimum and 2-d maximum coordinate is selected
to form the bounding box. 

To elaborate on finding the bboxes, contiguous regions are found using `scipy.ndimage.measurements.label()`. Each contiguous
region is enumerated by the function and a list of enumerations is the secound output of the function. After that, 
a selection of enumerations makes the location of the bboxs simple. 

The more sophisticated filter which the question is really asking about is in car_tracker.py. Two classes are employed.
One is a wrapper class which contains all detected cars. The second is a car class which contains the past 5 observations.
There are three important ideas used in these classes. The first is matching. if a bounding box is discovered,
it is matched to the closest previously observed car. If there are multiple matches to a car, the closest is accepted.

The second is that of entering regions. A car may only be discovered in two regions. 
Once it is 'discovered,', it enters a 'potential_car' queue where it must remain
for 5 frames. If it disappears for 3 frames during the wait period, it is deleted. 

The third idea is to smooth out the observations. First, new observations are compared to the average
of the last 5 observations. If the observation does not differ in distance of the two defining coordinates
by more than 15%, an observation is accepted. Then the observations are averaged.

I did some experimentation on the third idea including the physics-based smoothing I employed in the 
previous project. It turned out that the physics-based smoothing in its most naive form is not effective
due to the constant noise in the bounding boxes. 


### Here is one image with its detections and corresponding heatmap:

![alt text][image5]

### This is the resulting boudninb box from the same image:
![alt text][image6]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main weakness of this pipeline is the speed. As mentioned previously, it takes about 3 seconds to do each iteration. 
This can be fixed by trying two things: decreasing the number of patches to search through and decreasing the number
of features needed for the classifier. At this point, it's not clear to me if the issue arises from the classifier 
not being good enough to pick out the car unless it is in very specific positions(do we need different features?) or 
if it is about picking out the right way to step through the picture. 

On top of this, the approach that I used for smoothing and giving labels to newly appearing cars could be a brittle 
method. One example is how the framework breaks down if the bounding boxes merge. When this happens the algorithm
treats multiple cars as one item and deletes its knowledge that there were multiple cars. 

The way that all of this interacts with the 'entering region' could cause many problems. Say a car's region merges 
temporarily outside of the 'entering region', then the algorithm loses the capability of determining that
the car region unmerged and the algorithm will believe that one car simply disappeared. 
