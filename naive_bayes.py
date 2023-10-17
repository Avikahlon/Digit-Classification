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
        num_classes = len(np.unique(y_train))
        num_features = x_train.shape[1]
        feature_probs = np.zeros((num_classes, num_features)) # need to change the intialize
        "*** YOUR CODE HERE ***"
       
        for i in range(num_classes):
            num_instances_in_class_i = np.sum(y_train == i)
            for j in range(num_features):
                pass
                num_instances_with_feature_j = np.sum((y_train == i) & (x_train[:, j] == 1))
                feature_probs[i, j] = num_instances_with_feature_j / num_instances_in_class_i
                #feature_probs[i, j] = len(x_train[(y_train==i) & (x_train[:,j]==1)])/len(y_train[y_train==i])
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

    def predict(self, x_test):
        """
        Predict the class labels for test sample.
    
        Args:
        x_test: Test features.
    
        Returns:
        predictions: Predicted class labels for test features.
        """
        num_samples, num_features = x_test.shape
        classes = sorted(list(np.unique(y_train)))
        num_classes = len(self.class_probs)
        predictions = np.zeros(num_samples)
        "*** YOUR CODE HERE ***"
        for i, c in classes:
            prior = np.log(self.calculate_class_prob[i])
            posterior = np.sum(np.log(self.calculate_feature_probs()))
            posterior = posterior + prior
            predictions.append(posterior)

        return predictions

y_train = np.load("D:/ICT203/y_train.npy")
X_train = np.load("D:/ICT203/x_train.npy") 
X_train = X_train / 255
X_train = X_train.reshape(len(X_train), 28*28)

X_train[X_train >= 0.5] = 1
X_train[X_train < 0.5] = 0    
nb = NaiveBayes()
#print(nb.calculate_class_probs(y_train))
fb = nb.calculate_feature_probs(X_train, y_train)
print(fb.shape)