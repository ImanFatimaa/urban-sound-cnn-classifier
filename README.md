# Urban Sound Classification using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) for environmental sound classification using the UrbanSound8K dataset. The system converts raw audio signals into Mel-Spectrogram and MFCC features and classifies them into 10 sound categories using a deep learning model built in PyTorch.

---

## Project Overview

The goal of this project is to classify real-world environmental audio clips into predefined categories using deep learning. The pipeline includes audio preprocessing, feature extraction, data augmentation, CNN model training, and evaluation using standard classification metrics.

---

## Dataset

The project uses the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (<= 4 seconds) from 10 classes.

Classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

The dataset is organized into 10 folds for cross-validation.

---

## Feature Extraction

Two main audio feature representations are used:

1. Mel-Spectrogram
   - 128 Mel filter banks
   - Converted to decibel scale
   - Fixed size of 128 x 174

2. MFCC Features
   - 40 MFCC coefficients
   - Zero-padded or truncated to 174 time frames

Librosa is used for all audio processing.

---

## Data Augmentation

To improve generalization and reduce overfitting, the following augmentations are applied during training:

- Additive noise injection
- Pitch shifting
- Time stretching

---

## Model Architecture

A custom Convolutional Neural Network is implemented using PyTorch.

Architecture:

Input: (1 x 40 x 174)

- Conv2D (1 → 32), BatchNorm, ReLU
- MaxPooling
- Conv2D (32 → 64), BatchNorm, ReLU
- MaxPooling
- Conv2D (64 → 128), BatchNorm, ReLU
- MaxPooling
- Flatten
- Fully Connected Layer (128 * 5 * 21 → 256)
- Dropout (0.5)
- Output Layer (256 → 10 classes)

---

## Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Weight Decay: 1e-3
- Batch Size: 32
- Epochs: 30
- Scheduler: ReduceLROnPlateau
- Early Stopping: Patience = 5

---

## Training Strategy

The model is trained using:

- Training set: Folds 1–8
- Validation set: Fold 9
- Test set: Fold 10

Early stopping is applied based on validation accuracy. The best model is saved automatically as best_model.pth.

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1 Score (Weighted)
- Confusion Matrix

Seaborn heatmap is used to visualize the confusion matrix.

---

## Training Outputs

During training, the following are tracked and plotted:

- Training loss vs validation loss
- Training accuracy vs validation accuracy
- Learning rate changes

---

---

## Key Features

- PyTorch-based CNN implementation from scratch
- Mel-Spectrogram and MFCC feature extraction
- Data augmentation for robustness
- Early stopping to prevent overfitting
- Learning rate scheduling
- Full evaluation pipeline with metrics
- Confusion matrix visualization

---

## Future Improvements

- Replace CNN with CRNN (CNN + RNN hybrid)
- Use Transformer-based audio classification models
- Real-time sound classification system
- Deployment using Flask or FastAPI
- Optimization for edge devices and mobile deployment

---

## Author

This project was developed as part of a deep learning and audio signal processing study using the UrbanSound8K dataset.
