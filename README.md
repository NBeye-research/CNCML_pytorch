# **CNCML: a meta-learning framework for transferring keratitis detection from slit-lamp to smartphone utilizing limited smartphone Photographs**

## create time: 2024.01.15

## Introduction
This repository contains the source code for training an innovative model using cosine nearest centroid-based metric learning (CNCML) in scenarios with limited data availability. CNCML is designed to identify keratitis on smartphones in an automated fashion. This model effortlessly transitions from professional slit-lamp devices to portable devices like smartphones, delivering reliable performance for keratitis detection.

## prerequisities
* Python 3.8.13
* Pytorch 1.11.0
* tensorboardX
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5

This repository has been tested on NVIDIA RTX2080Ti. Configurations (e.g batch size, image size) may need to be changed on different platforms.

# Installation
Other packages are as follows:
* wheel: 0.41.2
* scipy: 1.10.1
* opencv-python: 4.8.1.78
* scikit-image: 0.18.0
* sikit-learn: 1.3.0
* numpy: 1.24.4
* tqdm: 4.64.1
* torchvision: 0.12.0
* yaml: 0.2.5

# Install dependencies
pip install -r requirements.txt

# Usage
* The file "train_classifier.py --config ./configs/train_classifier.yaml" in /CNCML-Source is used for feature extractor learning or traditional deep learning.
* The file "train_meta.py --config ./configs/train_meta.yaml" in /CNCML-Source is used for metric learning.
* The file "test_meta.py --config ./configs/test_meta.yaml" in /CNCML-Source is used for testing of CNCML.
* The file "test_classifier.py --config ./configs/test_classifier.yaml" in /CNCML-Source is used for testing of traditional deep learning model.

# Advanced instructions
The training parameters and model selection can be modified within the config file.
For example, in 'train_classifier.yaml,' you can select a different feature extractor structure by altering 'model_args/encoder'.

**Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Yangyang Wang, E-mail: youngwang666@hotmail.com.**
