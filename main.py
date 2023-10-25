# main.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

from argparse import ArgumentParser
import time
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from nb_data_loader import *
from alt_data_loader import *
from alt_model import ALTModel
from naive_bayes import NaiveBayes
import torch.nn as nn
from tqdm import tqdm

USAGE_STRING = """
  USAGE:      python main.py <options>
  EXAMPLES:   (1) python main.py --c nb --d digitdata --mode train
                  - trains the naive bayes classifier on the digit dataset
              (2) python main.py --classifier alt  --data_dir digitdata --mode train --batch_size 64 --epoch 5 --learning_rate 0.0001
                  - trains the alternative model
                  """

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def metrics(predicted, true):

  report = classification_report(true, predicted, target_names=classes)
  print (report)


def missclassified(X, predicted, y_true):
  misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != predicted[i]]

  # Select a subset of misclassified samples (e.g., the first 25)
  subset_indices = misclassified_indices[:25]

  # Create a grid to display the misclassified samples
  num_rows = 5
  num_cols = 5
  plt.figure(figsize=(10, 10))

  for i, index in enumerate(subset_indices):
    plt.subplot(num_rows, num_cols, i + 1)

    # Display the digit image (replace 'X' with your image data)
    plt.imshow(X[index].reshape(28, 28), cmap='gray')  # Example for MNIST-like data
    plt.title(f'True: {y_true[index]}\nPred: {predicted[index]}')  # Set titles for each subplot
    plt.axis('off')

  plt.suptitle("Examples of Misclassified Digits")
  plt.tight_layout()
  plt.show(block=False)


def roc(y_pred, y_true):
  n_classes = len(classes)
  y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))

  fpr = dict()
  tpr = dict()
  roc_auc = dict()

  for i in range(n_classes):

    y_true_class = [1 if label == i else 0 for label in y_true]
    y_pred_class = [1 if label == i else 0 for label in y_pred]

    fpr[i], tpr[i], _ = roc_curve(y_true_class, y_pred_class)
    roc_auc[i] = auc(fpr[i], tpr[i])

  plt.figure(figsize=(8, 6))
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange', 'pink', 'purple']  # Define colors for each class

  for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.2f})')

  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curves for Multi-Class Classification (0 to 9)')
  plt.legend(loc='lower right')
  plt.show(block=False)

def conf_matrix(y_pred, y_true):
  c_matrix = confusion_matrix(y_true, y_pred)
  cf_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=classes)
  cf_display.plot()
  plt.show()


if __name__ == "__main__":
  parser = ArgumentParser(USAGE_STRING)
  parser.add_argument('-c', '--classifier', help='The type of classifier', choices=['nb', 'alt'], required=True)
  parser.add_argument('-d', '--data_dir', help='the dataset folder name', type=str, required=True)
  parser.add_argument('-m', '--mode', help='train, val or test', type=str, required=True)
  parser.add_argument('-b', '--batch_size', help='batch size', type=int)
  parser.add_argument('-e', '--epoch', help='number of epochs', type=int)
  parser.add_argument('-l', '--learning_rate', help='learning rate', type=float)
  args = parser.parse_args()

  print("Doing classification")
  print("--------------------")
  print("classifier:\t" + args.classifier)

  if args.classifier == "nb":
    """
choose naive bayes
"""
    data = NBDataLoader(args.data_dir)
    data.x_train = data.x_train.reshape(len(data.x_train), 28 * 28)
    data.x_train = (data.x_train / 255)
    data.x_test = data.x_test.reshape(len(data.x_test), 28 * 28)
    data.x_test = (data.x_test / 255)

    model = NaiveBayes()
    model.train(data.x_train, data.y_train)
    predicted = model.predict(data.x_test)
    predicted = np.array(predicted)

    metrics(predicted.tolist(), data.y_test.tolist())
    missclassified(data.x_test, predicted.tolist(), data.y_test.tolist())
    roc(data.y_test.tolist(), predicted)
    conf_matrix(data.y_test.tolist(), predicted.tolist())

  else:
    """
choose the alternative model
"""
    model = ALTModel()
    data = ALTDataLoader(args.data_dir, args.mode)
    x = torch.tensor(data.x, dtype=torch.float32)
    x = x / 255
    x = x.reshape(len(x), 28 * 28)
    y = torch.tensor(data.y, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
      model.train()
      total_loss = 0

      for i in tqdm(range(0, len(x), args.batch_size), desc='Training'):
        optimizer.zero_grad()
        Xbatch = x[i:i + args.batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i + args.batch_size]
        loss = loss_fn(y_pred, ybatch.long())
        total_loss += loss
        loss.backward()
        optimizer.step()
      print(f'Finished epoch {epoch}, latest loss {total_loss}')

    y_pred = model(x)
    y_predicted_labels = [torch.argmax(i) for i in y_pred]
    y_predicted_labels = torch.tensor(y_predicted_labels)
    y_pred = y_predicted_labels.tolist()

    metrics(y_pred, data.y.tolist())
    missclassified(data.x, y_pred, data.y.tolist())
    roc(data.y.tolist(), y_pred)
    conf_matrix(data.y.tolist(), y_pred)

