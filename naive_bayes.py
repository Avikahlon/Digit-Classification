# naive_bayes.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


class NaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        1. smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        2. class_probs: the prior probabilities for each class.
        3. feature_probs: the conditional probabilities for each feature given the class.
        """
        self.classes = None
        self.num_classes = None
        self.class_probs = None
        self.feature_probs = None
        self.smoothing_factor = smoothing_factor

    def calculate_class_probs(self, y_train):
        """
        Calculate the prior probabilities for each class.

        Args:
        y_train: Training labels.
    
        Returns:
        class_probs: Array of prior probabilities for each class.
        """
        self.num_classes = len(np.unique(y_train))
        total_samples = len(y_train)
        classes, counts = np.unique(y_train, return_counts=True)
        class_probs = []

        "*** YOUR CODE HERE ***"

        for i in classes:
            class_count = np.sum(y_train == i)
            class_probs.append(class_count + self.smoothing_factor/total_samples + self.smoothing_factor)
        return class_probs

    def calculate_feature_probs(self, x_train, y_train):
        """
        Calculate the conditional probabilities for each feature given the class.
    
        Args:
        x_train: Training features.
        y_train: Training labels.
    
        Returns:
        feature_probs: Array of conditional probabilities for each feature and class.
        """
        self.classes = sorted(list(np.unique(y_train)))
        num_features = x_train.shape[1]
        feature_probs = []  # need to change the initialize
        "*** YOUR CODE HERE ***"

        for c in self.classes:
            index = y_train == c
            cdata = x_train[index]
            class_data = []
            for feature in range(num_features):
                pixel = cdata[:, feature]
                mean = np.mean(pixel)
                std = np.std(pixel)
                class_data.append((mean, std))
            feature_probs.append(class_data)
        return feature_probs

    def train(self, x_train, y_train):
        """
        Train the NaiveBayes classifier. Do not modify this method.
    
        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self.class_probs = self.calculate_class_probs(y_train)
        self.feature_probs = self.calculate_feature_probs(x_train, y_train)

    def Gaussian(self, x, mean, std):

        if std < 1e-6:
            dev = 1e-6
        else:
            dev = std

        exp_term = np.exp(-((x - mean) ** 2) / (2 * (dev ** 2)))
        return exp_term / (np.sqrt(2 * np.pi) * dev)

    def predict(self, x_test):
        """
        Predict the class labels for test sample.
    
        Args:
        x_test: Test features.
    
        Returns:
        predictions: Predicted class labels for test features.
        """
        "*** YOUR CODE HERE ***"
        predictions = []

        for x in tqdm(x_test, desc="Predicting"):
            result = self._predict(x)
            predictions.append(result)

        return predictions

    def _predict(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(1/self.class_probs[i])
            posterior = 0
            for feature, (mean, std) in enumerate(self.feature_probs[c]):
                if std == 0:  # Handle the case of zero standard deviation
                    continue
                likelihood = self.Gaussian(x[feature], mean, std)
                if likelihood == 0:
                    likelihood = 1
                posterior += np.log(likelihood)
            posterior = posterior + prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]