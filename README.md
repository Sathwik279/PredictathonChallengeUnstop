# Deepfake Detection with Convolutional Neural Networks (CNN)

This repository contains a deep learning model to detect deepfake images using a Convolutional Neural Network (CNN). The model is trained on a dataset of real and fake images and can predict whether an image is real or fake based on its features.

## Table of Contents

* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
    * [Train the Model](#train-the-model)
    * [Predict Deepfakes](#predict-deepfakes)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Prediction](#prediction)
* [Files in this Repository](#files-in-this-repository)
* [License](#license)

## Installation
Ensure that you have the necessary libraries installed. You can install the required dependencies by running:
pip install -r requirements.txt

## The requirements.txt file should include:
tensorflow
numpy
glob
pillow
scikit-learn

## Dataset
This project uses a dataset of real and fake images to train the model. You will need to download the dataset and specify the path to the following directories:
real_cifake_images: Folder containing real images.
fake_cifake_images: Folder containing fake images.
test: Folder containing test images for final evaluation.
Please update the paths in the code to match your local or cloud directories.
##Usage
### 1. Train the Model
To train the model on your dataset, run the following script:
python train_model.py
This script: 
Loads images from the real and fake directories.
Preprocesses the images by resizing them to 128x128 and normalizing the pixel values.
Splits the dataset into training and testing subsets.
Defines and compiles a CNN model.
Trains the model on the training set and evaluates it on the testing set.
Saves the trained model to disk for later use.
### 2.Predict Deep Fakes
To predict if images are real or fake, use the trained model. You can run the following script:
python predict.py
This script:
Loads the saved model.
Loads images from the test dataset.
Runs predictions on the images.
Saves the predictions as a JSON file in the format:
[
  {"index": 1, "prediction": "real"},
  {"index": 2, "prediction": "fake"},
  ...
]

## Model Training
The CNN model architecture is built with multiple convolutional and max-pooling layers to capture hierarchical features. The model is compiled with the Adam optimizer and binary cross-entropy loss. Dropout layers are added to avoid overfitting during training.
After training on 80% of the dataset, the model is evaluated on the remaining 20% of the data. The trained model is then saved as a .keras file for future predictions.

## Evaluation
After training the model, you can evaluate it using the test dataset to see how well the model generalizes. The evaluation provides the test accuracy and loss.

## Prediction
Once the model is trained, you can use it to predict deepfake images by running the predict.py script. It will output a JSON file with predictions.

## Files in the directory
train_model.py: Script to train the CNN model.
predict.py: Script to run predictions on test images.
model.train80.keras: Model trained on 80% of the dataset.
model.fullTrain.keras: Final model trained on the entire dataset.
requirements.txt: List of dependencies for the project.
fix errors
