# Image Classification on Food-101 Dataset

This project involves implementing and evaluating various image classification models using the Food101 dataset. The models include a simplified version of the VGG architecture, as well as utilizing pre-trained models like ResNet and VGG from PyTorch for accurate food image classification.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Implemented Models](#implemented-models)
- [Acknowledgements](#acknowledgements)

## Introduction

In this project, we explore different approaches to image classification using the Food101 dataset. We'll implement a range of models and evaluate their accuracy in classifying various food items. The models are implemented using PyTorch, and both custom architectures and pre-trained models are utilized.

## Requirements

To run this project, you'll need the following:

- Python (>=3.6)
- PyTorch (>=1.0)
- TorchVision (>=0.2)
- Matplotlib (for visualizations)

## Usage

## Implemented Models

1. Custom VGG Model: A simplified version of the VGG architecture is implemented for food image classification. This model is defined in the train_custom_vgg.py script.
2. Pre-trained ResNet Model: A pre-trained ResNet model from PyTorch's torchvision models is used for food image classification. This model is implemented in the train_pretrained_resnet.py script.
3. Pre-trained VGG Model: A pre-trained VGG model from PyTorch's torchvision models is used for food image classification. This model is implemented in the train_pretrained_vgg.py script.
Feel free to modify the scripts, parameters, and settings to experiment with different model architectures and training configurations.

## Acknowledgements
The models in this project are inspired by various deep learning architectures and the PyTorch library. The Food101 dataset is acknowledged as the source of data for evaluation purposes. This project is developed by Winson for the purpose of learning and exploring image classification techniques.
