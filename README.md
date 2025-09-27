# Handwritten Digit Classification: Naïve Bayes vs. Feed-Forward Neural Network
## Project Overview
This project focuses on the classification of handwritten digits (0-9) using a dataset of 28×28 pixel images. The primary objective is to compare the performance, strengths, and weaknesses of two distinct machine learning models:

Gaussian Naïve Bayes (GNB): Chosen for its simplicity and efficiency.

Feed-Forward Neural Network (FNN): Chosen as a modern, common deep learning approach for comparison.

The goal is to determine which classifier is more effective for this specific use case, especially considering the inherent complexity and inconsistency in handwritten digit styles and sizes within the provided image data.

## Data
The dataset consists of images, each sized 28×28 pixels, paired with a label from 0 to 9. The significant variability in the style, size, and slant of the handwritten digits makes this a challenging classification problem.

## Model Architecture and Implementation
### Model 1: Gaussian Naïve Bayes
The Naïve Bayes classifier assumes feature independence (a strong assumption for image data) and uses a Gaussian distribution to model the conditional probability of features given a class.

Smoothing: Uses Laplace smoothing to handle zero probabilities and ensure robust calculations of class priors.

calculate_feature_probs: Calculates the mean and standard deviation of feature values (pixel intensities) for each class, assuming a Gaussian distribution.

Prediction: Uses the standard Naïve Bayes formula to compute the posterior probability and selects the class with the highest probability.

### Model 2: Feed-Forward Neural Network (FNN)
The FNN is built using the PyTorch library and designed for simplicity and speed.

Input Layer:784 input features (from the flattened 28×28 image).

Hidden Layer 1: Linear layer with 784 inputs and 392 outputs. Weights initialized using He (Kaiming Uniform) initialization.

Activation 1: ReLU (Rectified Linear Unit).

Hidden Layer 2: Linear layer with 392 inputs and 128 outputs.

Activation 2: ReLU.

Output Layer: Linear layer with 128 inputs and 10 outputs (representing the 10 digit classes).

Training: Involves training based on tunable hyperparameters like learning rate, epochs, and batch size.

## Evaluation and Results
### Evaluation Methods
Model performance was evaluated based on:

Quantitative Metrics: F1 Score, Accuracy, Precision, Recall, Weighted Average, and Macro Average (calculated per class).

Visualizations: Confusion Matrix, ROC Curve, and a display of misclassified labels.

### Conclusion
The Feed-Forward Neural Network is far superior to the Gaussian Naïve Bayes model in both speed and accuracy.

|Metric | Gaussian Naïve Bayes (GNB) | Feed-Forward Neural Network (FNN) | Rationale|
|-------|----------------------------|-----------------------------------|----------|
|Accuracy|Lower (e.g., ~85%)|Significantly Higher|FNN can learn non-linear relationships and feature hierarchies, overcoming GNB's limiting assumption of feature independence.|
|Prediction Speed|Very Slow (~2-3 minutes for 10,000 samples)|Very Fast|FNN leveraging the PyTorch library benefits from GPU support and parallel processing/multithreading capabilities, drastically reducing computation time.|
|Per-Class Score|Highly variable (e.g., lower scores for complex digits like 5 and 9)|More consistent||

## Areas for Improvement
The following techniques could further enhance model performance, particularly for GNB:

Feature Modification: Implement dimensionality reduction (e.g., PCA) or feature selection to potentially improve GNB's performance, despite its independence assumption.

GNB Ensemble Methods: Utilize techniques like bagging or boosting to combine predictions from multiple GNB models.

Cross-Validation: Implement cross-validation for more robust hyperparameter tuning and performance assessment.

Advanced NN Models: Switch to Convolutional Neural Networks (CNNs), which are the standard, state-of-the-art approach specifically designed for learning spatial hierarchies in image data.
