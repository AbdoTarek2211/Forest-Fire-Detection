Fire Detection using Convolutional Neural Networks (CNN)

This repository contains code for a Fire Detection system implemented using Convolutional Neural Networks (CNNs). The system is designed to classify images as either containing fire or not containing fire.
Table of Contents

    Introduction
    Requirements
    Usage
    File Structure
    Results
    License

Introduction

The goal of this project is to develop a CNN-based model capable of accurately detecting fire in images. The dataset used for training and testing the model consists of images categorized into two classes: 'fire' and 'non-fire'. The CNN model is trained on this dataset to learn the distinguishing features between fire and non-fire images.
Requirements

To run the code, you need to have the following dependencies installed:

    Python 3.x
    TensorFlow
    Keras
    Pandas
    NumPy
    Matplotlib
    Seaborn
    OpenCV
    Plotly
    Scikit-learn

Usage

    Clone the repository to your local machine.
    Ensure all the necessary dependencies are installed (see Requirements).
    Run the fire_detection.py script.

File Structure

    fire_detection.py: Main Python script containing the code for fire detection using CNNs.
    README.md: Documentation file providing an overview of the project, usage instructions, and other relevant information.
    requirements.txt: Text file listing all the Python dependencies required to run the code.
    fire_dataset: Directory containing the dataset with two subdirectories: fire_images and non_fire_images, each containing respective images.

Results

Upon running the fire_detection.py script, the CNN model is trained on the dataset, and its performance is evaluated. Additionally, the script allows for predicting whether a given input image contains fire or not.

Fire Dataset

(https://www.kaggle.com/datasets/phylake1337/fire-dataset)
