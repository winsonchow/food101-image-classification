# Image Classification on Food-101 Dataset
## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Implemented Models](#implemented-models)
- [Acknowledgements](#acknowledgements)

## Introduction

This project involves implementing and evaluating various image classification models using the Food101 dataset. This repository contains code and information for building and evaluating food image classification models using PyTorch. The goal of this project is to develop and train deep learning models to classify images of various foods based on their categories. The project covers various aspects of deep learning model development, including data preprocessing, model architecture, training, evaluation, and visualization.

## Requirements

To run this project, you'll need the following:

- Python (>=3.6)
- PyTorch (>=1.0)
- TorchVision (>=0.2)
- Matplotlib (for visualizations)

## Overview

The project is structured around several key components:

1. Data Preparation: The data_setup.py module provides functions for setting up the dataset, creating data loaders, and applying transformations to the images.
2. Model Architecture: The project includes implementations of different model architectures, including SimpleVGG and pretrained models like ResNet and EfficientNet. These models are defined in separate modules.
3. Training and Evaluation: The engine.py module contains functions for training and evaluating the models. It includes functions for training a model on a dataset, evaluating its performance, and saving the best models.
4. Utilities: The project includes utility functions (utils.py) for plotting evaluation metrics, loading/saving models, and other common tasks.
5. Notebooks: Jupyter notebooks provide step-by-step guidance on data preprocessing, model creation, training, and evaluation.
6. Experiment Tracking: The project emphasizes good coding practices, including parameterization of experiments using configuration files or scripts.

## Implemented Models

1. Custom VGG Model: A simplified version of the VGG architecture is implemented for food image classification.
2. Pre-trained DenseNet Model: A pre-trained DenseNet model from PyTorch's torchvision models is used for food image classification.
3. Pre-trained ResNet Model: A pre-trained ResNet model from PyTorch's torchvision models is used for food image classification.
4. Pre-trained EfficientNet Model: A pre-trained EffNet model from PyTorch's torchvision models is used for food image classification.

## Acknowledgements
The models in this project are inspired by various deep learning architectures and the PyTorch library. The Food101 dataset is acknowledged as the source of data for evaluation purposes. This project is developed by Winson for the purpose of learning and exploring image classification techniques.
