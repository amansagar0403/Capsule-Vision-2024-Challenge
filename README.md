# Capsule-Vision-2024-Challenge


## Project Title: Automated Classification of Video Capsule Endoscopy Abnormalities

### Project Description
The purpose of this project is to develop a deep learning model that automatically classifies abnormalities detected in video capsule endoscopy (VCE) frames. The model focuses on ten specific abnormalities, including angioectasia, bleeding, and ulcers, with the aim of assisting healthcare professionals in diagnosing gastrointestinal disorders more efficiently.

## Usage Instructions

To run the project, use the following command in your terminal:

```bash
python FinalSubmission.py
```
## Dataset Information

The dataset consists of video frames from capsule endoscopy procedures, containing various gastrointestinal abnormalities.

### Source
- The dataset is a combination of publicly available datasets and private collections, specifically curated for this competition.
- [Training and Validation Dataset of Capsule Vision 2024 Challenge](https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469?file=48018562)


### Structure
- The dataset is organized into training and validation directories, with subdirectories representing each class of abnormalities.

### Preprocessing Steps
- Images are resized to a uniform dimension.
- Data augmentation techniques are applied to improve model robustness.

## Model Architecture

The model is built using a hybrid architecture that combines the strengths of ResNet and DenseNet.

### Layers
- Convolutional layers for feature extraction.
- Batch normalization and dropout layers to prevent overfitting.
- A fully connected layer customized for multi-class classification.

This architecture allows the model to capture complex features necessary for identifying the abnormalities in the video frames.

## Results

Results are logged during the training process, including metrics such as accuracy, loss, and area under the curve (AUC).


## Acknowledgments

- Capsule Vision Competition organizers for providing the dataset and competition platform.
- Various open-source libraries and frameworks (e.g., TensorFlow, PyTorch) that facilitated the development of this project.

