## Image Caption Generator

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
| <img width="50%" src="https://drive.google.com/drive/folders/154ObbxDt5m7wI4ZgXwOYlNztskwzIEVK?usp=sharing/img_4.jpg" alt="Image 1"> | <ul><li><strong>Greedy:</strong> A man in a blue shirt is riding a bike on a dirt path.</li><li><strong>BEAM Search, k=3:</strong> A man is riding a bicycle on a dirt path.</li></ul>|
| <img src="https://github.com/dabasajay/Image-Caption-Generator/raw/master/test_data/surfing.jpeg" alt="Image 2"> | <ul><li><strong>Greedy:</strong> A man in a red kayak is riding down a waterfall.</li><li><strong>BEAM Search, k=3:</strong> A man on a surfboard is riding a wave.</li></ul>|

## 4. Procedure to Train Model

1. Clone the repository to preserve directory structure.<br>
`git clone https://github.com/dabasajay/Image-Caption-Generator.git`
2. Put the required dataset files in `train_val_data` folder (files mentioned in readme there).
3. Review `config.py` for paths and other configurations (explained below).
4. Run `train_val.py`.

## 5. Procedure to Test on new images

1. Clone the repository to preserve directory structure.<br>
`git clone https://github.com/dabasajay/Image-Caption-Generator.git`
2. Train the model to generate required files in `model_data` folder (steps given above).
3. Put the test images in `test_data` folder.
4. Review `config.py` for paths and other configurations (explained below).
5. Run `test.py`.

## 6. Configurations (config.py)

**config**

1. **`images_path`** :- Folder path containing flickr dataset images
2. `train_data_path` :- .txt file path containing images ids for training
3. `val_data_path` :- .txt file path containing imgage ids for validation
4. `captions_path` :- .txt file path containing captions
5. `tokenizer_path` :- path for saving tokenizer
6. `model_data_path` :- path for saving files related to model
7. **`model_load_path`** :- path for loading trained model
8. **`num_of_epochs`** :- Number of epochs
9. **`max_length`** :- Maximum length of captions. This is set manually after training of model and required for test.py
10. **`batch_size`** :- Batch size for training (larger will consume more GPU & CPU memory)
11. **`beam_search_k`** :- BEAM search parameter which tells the algorithm how many words to consider at a time.
11. `test_data_path` :- Folder path containing images for testing/inference
12. **`model_type`** :- CNN Model type to use -> inceptionv3 or vgg16
13. **`random_seed`** :- Random seed for reproducibility of results

**rnnConfig**

1. **`embedding_size`** :- Embedding size used in Decoder(RNN) Model
2. **`LSTM_units`** :- Number of LSTM units in Decoder(RNN) Model
3. **`dense_units`** :- Number of Dense units in Decoder(RNN) Model
4. **`dropout`** :- Dropout probability used in Dropout layer in Decoder(RNN) Model

## 7. Frequently encountered problems

- **Out of memory issue**:
  - Try reducing `batch_size`
- **Results differ everytime I run script**:
  - Due to stochastic nature of these algoritms, results *may* differ slightly everytime. Even though I did set random seed to make results reproducible, results *may* differ slightly.
- **Results aren't very great using beam search compared to argmax**:
  - Try higher `k` in BEAM search using `beam_search_k` parameter in config. Note that higher `k` will improve results but it'll also increase inference time significantly.

## 8. TODO

- [X] Support for VGG16 Model. Uses InceptionV3 Model by default.

- [X] Implement 2 architectures of RNN Model.

- [X] Support for batch processing in data generator with shuffling.

- [X] Implement BEAM Search.

- [X] Calculate BLEU Scores using BEAM Search.

- [ ] Implement Attention and change model architecture.

- [ ] Support for pre-trained word vectors like word2vec, GloVe etc.

## 9. References

<ul type="square">
	<li><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a> - Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan</li>
	<li><a href="https://arxiv.org/abs/1703.09137">Where to put the Image in an Image Caption Generator</a> - Marc Tanti, Albert Gatt, Kenneth P. Camilleri</li>
	<li><a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">How to Develop a Deep Learning Photo Caption Generator from Scratch</a></li>
</ul>
