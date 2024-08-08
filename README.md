# traffic
### Background
Implementation of CS50AI Week 5 - Traffic. Built a neural
network to classify road signs from the German Traffic
Sign Recognition Benchmark (GTSRB) dataset. Used LeNet 
structure described in open-source textbook Dive into Deep
Learning.

### Files
traffic.py: functions to load and display data + train model\
model.py: LeNet implementation\
custom_dataset.py: custom Dataset to fit PyTorch DataLoader\
classify.py: allows user to submit an image for classification

### Usage
Download the [dataset](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip)\
Train and (optionally) save a model: python traffic.py data_directory [saved_model_name]\
Test a trained model on user-submitted image: python classify.py saved_model_name image_path

