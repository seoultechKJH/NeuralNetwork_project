import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from imblearn.over_sampling import SMOTE

dataset = pd.read_csv('project_dataset.csv')

from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

x_train_set = dataset.drop('산업분야', axis=1)
x_train = min_max_scaler.fit_transform(x_train_set)
x_train = pd.DataFrame(x_train, columns = x_train_set.columns.tolist())
print(x_train.head())
x_train = x_train.values

y_train = (dataset['산업분야']).values

print(Counter(y_train))
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.shape(x_train))
print(np.shape(y_train))

x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))

def accuracy(y_hat, y):
    count = 0
    for i in range(len(y)):
        if(y_hat[i] == y[i]):
            count = count+1
    return count/len(y)

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

import torch.nn as nn
from torch import optim

class FirstNetwork_v3(nn.Module):
  
  def __init__(self):
    super().__init__()
    torch.manual_seed(0)
    self.net = nn.Sequential(
        nn.Linear(9, 64), 
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4), 
        nn.Softmax()
    )

  def forward(self, X):
    return self.net(X)

  def predict(self, X):
    Y_pred = self.forward(X)
    return Y_pred

history = []

def fit_v2(x, y, model, opt, loss_fn, epochs = 50000):
  
  for epoch in range(epochs):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    history.append(loss.item())
  return loss.item()

fn = FirstNetwork_v3()
loss_fn = F.cross_entropy
opt = optim.Rprop(fn.parameters(), lr=0.001)
fit_v2(x_train, y_train, fn, opt, loss_fn)
loss = history[-1]
print('Final loss', loss)

plt.plot(history)
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sn

y_pred = [fn.predict(sample).tolist().index(max(fn.predict(sample).tolist())) for sample in x_test]
y_true = [sample.item() for sample in y_test]

classes = ('0', '1', '2', '3')

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.show()

print(accuracy(y_pred, y_true))
