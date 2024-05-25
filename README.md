# Animal Image Classification

This project is a simple image classification project using EfficientNetV2 to classify images of animals. The dataset used in this project is the [Kaggle Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals). The dataset contains 90 different animals with 60 images per class.

## Requirements

- Run the following command to install the required packages:
  `pip install -r requirements.txt`
- Cuda 12.1 is required to run the code on GPU.

## Usage

- Run the following command to train the model without pre-trained weights (EfficientNetV2):  
  `python train.py`
- Run the following command to train the model with pre-trained weights (EfficientNetV2):  
  `python pretrained.py`
- Run the following command to test the model after training:  
  `python test_predict.py`

## Results

The model trained without pre-trained weights achieved an best accuracy of 0.453 on the test set after 200 epochs.  
The model trained with pre-trained weights achieved an best accuracy of 0.953 on the test set.
