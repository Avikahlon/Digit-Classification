# naive_bayes.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np
from tqdm import tqdm
import icecream as ic

class NaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        1. smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        2. class_probs: the prior probabilities for each class.
        3. feature_probs: the conditional probabilities for each feature given the class.
        """
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
        num_classes = len(np.unique(y_train))
        classes = sorted(list(np.unique(y_train)))
        class_probs = np.zeros(num_classes)
        "*** YOUR CODE HERE ***"
        for i in classes:
            prob = len(y_train[y_train==i])/len(y_train)
            class_probs[i] = prob

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
        self.num_classes = len(np.unique(y_train))
        num_features = x_train.shape[1]
        feature_probs = np.zeros((self.num_classes, num_features, 2)) # need to change the intialize
        "*** YOUR CODE HERE ***"

        for feature in range(num_features):
            feature_values = x_train[:, feature]
            feature_params = {}
            for class_label in np.unique(y_train):
                samples_in_class = x_train[y_train == class_label]
                mean = np.mean(samples_in_class[:, feature])
                std = np.std(samples_in_class[:, feature])
                feature_probs[class_label, feature, 0] = mean
                feature_probs[class_label, feature, 1] = std


        def get_feature_probs(feature_index):
            return feature_probs[feature_index]

        return get_feature_probs


    def train(self, x_train, y_train):
        """
        Train the NaiveBayes classifier. Do not modify this method.
    
        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self.class_probs = self.calculate_class_probs(y_train)
        self.feature_probs = self.calculate_feature_probs(x_train, y_train)

    def predict(self, x_test):
        """
        Predict the class labels for test sample.
    
        Args:
        x_test: Test features.
    
        Returns:
        predictions: Predicted class labels for test features.
        """
        num_samples = x_test.shape[0]
        num_features = x_test.shape[1]
        predictions = np.zeros(num_samples)
        "*** YOUR CODE HERE ***"
        for i in range(num_samples):
            posteriors = np.zeros(self.num_classes)

            for c in range(self.num_classes):
                posterior = np.log(self.class_probs[c])

                for feature in range(x_test.shape[1]):
                    mean, std = self.feature_probs[c, feature, 0], self.feature_probs[c, feature, 1]
                    if std == 0:  # Handle the case of zero standard deviation
                        continue
                    likelihood = self._pdf(x_test[i, feature], mean, std)
                    posterior += np.log(likelihood)

                posteriors[c] = posterior

            predictions[i] = np.argmax(posteriors)

        return predictions