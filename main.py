# main.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

from argparse import ArgumentParser

import torch.optim

from nb_data_loader import *
from alt_data_loader import *
from alt_model import ALTModel
from torch.utils.data import DataLoader
from naive_bayes import NaiveBayes
import time
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

USAGE_STRING = """
  USAGE:      python main.py <options>
  EXAMPLES:   (1) python main.py --c nb --d digitdata --mode train
                  - trains the naive bayes classifier on the digit dataset
              (2) python main.py --classifier alt  --data_dir digitdata --mode train --batch_size 64 --epoch 5 --learning_rate 0.0001
                  - trains the alternative model
                  """
  

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
    x_train = data.x_train / 255
    x_train = x_train.reshape(len(x_train), 28 * 28)
    x_test = data.x_test / 255
    x_test = x_test.reshape(len(x_test), 28 * 28)
    nb = NaiveBayes()
    nb.train(data.x_train, data.y_train)
    predicted = nb.predict(data.x_test)
    accuracy = (predicted == data.y_test)
    print(accuracy)

  else:
    """
    choose the alternative model
    """
    alt = ALTModel()
    data = ALTDataLoader(args.data_dir, args.mode)
    x = torch.tensor(data.x, dtype=torch.float32)
    x = x / 255
    x = x.reshape(len(x), 28 * 28)
    y = torch.tensor(data.y, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(alt.parameters(), lr=args.learning_rate)
    #print(data.y.shape)

    for epoch in range(args.epoch):
      alt.train()
      total_loss = 0
      # print(model.hidden1.weight)
      for i in range(0, len(x), args.batch_size):
        optimizer.zero_grad()
        Xbatch = x[i:i + args.batch_size]
        y_pred = alt(Xbatch)
        ybatch = y[i:i + args.batch_size]
        loss = loss_fn(y_pred, ybatch.long())
        total_loss += loss
        loss.backward()
        optimizer.step()
      print(f'Finished epoch {epoch}, latest loss {total_loss}')

    y_pred = alt(x)
    y_predicted_labels = [torch.argmax(i) for i in y_pred]
    y_predicted_labels = torch.tensor(y_predicted_labels)
    # print(y_predicted_labels.size(), y_train.size())
    accuracy = (y_predicted_labels == y).float().mean()
    print(f"Accuracy {accuracy * 100}")


     

     



