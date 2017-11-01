# Traffic_SignClassifier
German Traffic Sign CNN network
**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/label_frequency_histogram.png "Visualization"
[image4]: ./examples/training_colored_images1.png 
[image2]: ./examples/stop_sign_color.png  "color stop sign"
[image5]: ./examples/stop_sign_gray.png  "gray stop sign "
[image3]: ./examples/new_test_results.png "softmax prob. results"
[image6]: ./examples/final_network_arch.jpg "final network arch"
[image7]: ./new-traffic-signs/1.png
[image8]: ./new-traffic-signs/2.png
[image9]: ./new-traffic-signs/3.png
[image10]: ./new-traffic-signs/4.png
[image11]: ./new-traffic-signs/5.png
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset: 

###### Training data bar chart 

![alt text][image1]

###### 5-Random Samples of the training set   

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this will give us a better accuracy than the colored ones, Also the color images  doesn't add much in the model training

As a last step, I normalized the image data because it makes it easier for optimization so I normalized using this formula: 
  
(X_train - mean(X_train)) / (max(X_train) - min(X_train))

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]    ![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image6]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scaled image   					| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU                  |                                               |
| dropout				| 										        |
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU                  |                                               |
| dropout				| 										        |
| Convolution 5x5 	    | 1x1 stride, valid padding, outputs 20x20x26 	|
| RELU	                |            									|
| Max pooling   		| 2x2 stride, outputs 10x10x26           		|
| dropout				|            									|
| Convolution 5x5 	    | 1x1 stride, valid padding, outputs 6x6x36 	|
| RELU	                |            									|
| Max pooling   		| 2x2 stride, outputs 3x3x36           		    |
| dropout				|            									|
| Flatten				| Flatten the output of conv4, output 1x324		|
| fully conneted layer 	| input 1x324, output 1x120						|
| RELU					|												|
| dropout				|    											|
| fully conneted layer 	| input 1x120, output 1x84						|
| RELU					|												|
| dropout				|    											|
| fully conneted layer 	| input 1x84, output 1x43 						|
|    					|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

1. First I have used the LeNet arch. after applying data pre-processing step 
(converting the image to gray scale then normalizing it with min-max tech) and after several experiments I have reached to validation accuracy 90% with these hyperparameters: epochs = 30 , learning rate = 0.001 and patch size = 128, Although the training accuracy was almost 97%.
2. I applied a modification to the LeNet arch. to over come the overfitting of the training data. This modification had a fair increase for the accuracy to be 94% with the same hyperparameter in the above step.
3. I have tried to tune the hyperparamters to increase both training and validation accuracy but with no hope the accuracy stuck to 94%
4. So I went to change in the number of conv layers in the LeNet network, So my first modification trail gives a slight enhance in the validation accuracy which becomes 96% with these hyperparameters: epochs = 50 , learning rate = 0.001 and the patch size = 128. 
5. Then I have tried to decrease the patch size to be 78 instead of 128 , As the small patch size will produces choppier, more stochastic weight updates. And the results increased to 97% for the validation accuracy.
6. Then I have made again another modification in the number of conv layers and in the filters slice depths to make network able to extract more high level features and enhance the network learning for the traffic signs shapes, So with the same previous hyperparameters this arch. gives a better accuracy results for the training it was 100% and for the validation set it was 98.5%  and this was my final arch.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.5%
* test set accuracy of 96.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

1. First I have started with the LeNet Arch. as a quick start point.
2. The validation accuracy was low with respect to the training accuracy; the accuracy was 90% for validation and 97% for the training.
3. So I thought that I should add a dropout layers to the LeNet to reduce from the over-fitting of the training data by adding a dropouts layers after each conv and full connected layers. This approach had a fair increase for the accuracy to be 94%.
4. Then I have added another conv layer connected before the first conv layer in the LeNet arch. and this was my 2nd architecture. This new arch. gives a better results for the training and validation accuracy it was almost 96% with these hyperparameters: epochs = 50 , learning rate = 0.001 and the patch size = 128.
5. Then I have added another conv layer connected to the above layer to make the network able to extract more high level features and enhance the learning for the traffic signs shapes and this was my 3rd arch.
6. The last arch. with the same previous hyperparameters gives a better accuracy results for the training it was 100% and for the validation set it was 98.5%  and this was my final arch. That I have described in the above diagram.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

The images are normal but there are varieties in colors which can cause issue with the model and this point need to be tested.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  			         	| 
| Turn left ahead	    | Turn left ahead	 							|
| General Caution	    | General Caution				                |
| Road work		      	| Road work						 			    |
| Speed limit (60km/h)	| Speed limit (60km/h)      				    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This more than the accuracy of the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit (30km/h)  						| 
| 1.     				| Turn left ahead                               |
| 1.					| General Caution							    |
| 1.	      			| Road work				 				        |
| 1.				    | Speed limit (60km/h)   						|


This image shows the certainty of model predictions, this is the best 3 predictions for each input.


![alt text][image3]
