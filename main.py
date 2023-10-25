# main.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn import metrics
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

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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

    f1_score = round(metrics.f1_score(data.y_test, predicted, average='weighted'), 10)
    recall_score = round(metrics.recall_score(data.y_test, predicted, average='weighted'), 10)
    accuracy_score = metrics.accuracy_score(data.y_test, predicted)
    print(f'F1 Score: {f1_score}')
    print(f'Recall Score: {recall_score}')
    print(f'Accuracy Score: {accuracy_score*100}')

    confusion_matrix = metrics.confusion_matrix(data.y_test.tolist(), predicted.tolist())
    cf_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cf_display.plot()
    plt.show()

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

    f1_score = round(metrics.f1_score(data.y.tolist(), y_pred, average='weighted'), 10)
    recall_score = round(metrics.recall_score(data.y.tolist(), y_pred, average='weighted'), 10)
    accuracy_score = metrics.accuracy_score(data.y.tolist(), y_pred)
    print(f'F1 Score: {f1_score}')
    print(f'Recall Score: {recall_score}')
    print(f'Accuracy Score: {accuracy_score*100}')

    confusion_matrix = metrics.confusion_matrix(data.y.tolist(), y_pred)
    cf_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    cf_display.plot()
    plt.show()
