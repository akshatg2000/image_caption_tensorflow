# Image Caption Generator

A neural network to generate captions for an image using CNN and RNN with BEAM as well as Greedy Search.


# Content ->

## 1. Requirements 

Recommended System Requirements to train model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 8GB memory</li>
	<li>Atleast 8GB of RAM</li>
	<li>Active internet connection </li>
</ul>

## 2. Installation

<u>Required libraried</u> - 

<ul type="square">
  <li>Numpy - 1.16.4</li>
	<li>Python - 3.6.7</li>
  <li>Keras - 2.2.4</li>
	<li>Tensorflow - 1.13.1</li>
	<li>nltk - 3.2.5</li>
	<li>PIL - 4.3.0</li>
	<li>Matplotlib - 3.0.3</li>
	<li>tqdm - 4.28.1</li>
</ul>

DataFile Required - Download from <a href="https://drive.google.com/drive/folders/1uEn7NHxYDKBD07IestKXthw-p3je-cQx?usp=sharing">link</a></li>

<ul type="square">
	<li>Flickr8k_Dataset:   contain images</li>
  <li>Flickr8k.token.txt: contain 5 caption for each token or imageID</li>
  <li>Flickr8k.trainImages.txt: contain imageId of train images</li>
  <li>Flickr8k.testImages.txt: contain imageId of test images</li>
</ul>




## 3. Generated Captions on Test Images

**Model used** - *InceptionV3 + LSTM*

| Image | Caption |
| :---: | :--- |
| <img width="60%" src="https://github.com/saket349/ImageCaptionGenerator/blob/5fd9d854074768381599b192a84fc95b1c96897b/WhatsApp%20Image%202021-05-08%20at%208.33.56%20PM.jpeg" alt="Image 1"> | <ul> <li><strong>Greedy:</strong> a football player in a red jersey is tackling another player in white who is tackling the ball.</li><li><strong>BEAM Search, k=3:</strong> a football player in a red jersey is tackling another player in red who is running with the ball whilst fans watch.</li><li><strong>BEAM Search, k=5:</strong> three football players are tackling a football player in a red and white uniform.</li><li><strong>BEAM Search, k=7:</strong> an american footballer in a red and white uniform gets ready to tackle an opposing player.</li><li><strong>BEAM Search, k=10:</strong> an american footballer in a red and white uniform gets ready to tackle an opposing player while fans watch.</li></ul>|
| <img src="https://github.com/saket349/ImageCaptionGenerator/blob/644a57dd20c1838dd5aee112a9ba3baa581dc818/WhatsApp%20Image%202021-05-08%20at%208.36.39%20PM.jpeg" alt="Image 2"> | <ul><li><strong>Greedy:</strong> a man in a red shirt climbing a rock.</li><li><strong>BEAM Search, k=3:</strong> a man in a red shirt climbing a rock.</li><li><strong>BEAM Search, k=5:</strong> a man climbing a rock.</li><li><strong>BEAM Search, k=7:</strong> a man climbing a rock.</li><li><strong>BEAM Search, k=10:</strong> a rock climber scales a steep rock cliff.</li></ul>|

## 4. Procedure to Train Model
```
In token_path, img_path, train_path, test_path & glove_path variable add
the path of Flickr8k.token.txt, Flicker8k_Dataset, Flickr_8k.trainImages.txt,
Flickr_8k.testImages.txt & glove file respectively
```

example 

```
token_path = '/content/drive/MyDrive/DS303/Flickr8k.token.txt'
img_path   = '/content/drive/MyDrive/DS303/Flicker8k_Dataset/'
train_path = '/content/drive/MyDrive/DS303/Flickr_8k.trainImages.txt'
test_path  = '/content/drive/MyDrive/DS303/Flickr_8k.testImages.txt'
glove_path = '/content/drive/MyDrive/DS303/glove.6B.200d.txt'

```
then run <b> .py</b> file in any preferable ide to train model, and if working on notebook run all cell to train and produce sample test result. 


## 5. Procedure to Test on images
For testing any image from the test data set -
- pick any image id of your choice from Flickr_8k.testImages.txt
- encode the image using encoding_test function
```
image = encoding_test[pic].reshape((1,2048))
```
- Now to get result 
 1. using greddy search
 ```
 greedySearch(image)
 ```
 2. Using Beam Search
 ```
 beamSearch_predictions(image, beam_index = 3)
 ```


## 6. To View Result access the given link below

<a href="https://drive.google.com/drive/folders/1Pmg3iOggAO56p_KOwfCQrJ0GYMfnbrQT?usp=sharing">Link</a> 
- Here you can find txt file of both greedy and beam search results
- the txt file contain prediction across each imageID


