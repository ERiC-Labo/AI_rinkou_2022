# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import datetime
import sys
import torch.quantization
import process as pc
import dataset as ds
import model

def convert_label_to_onehot(labels):
    onehot = np.zeros((len(labels), max(labels)+1))
    idx = [(i, t.item()) for i, t in enumerate(labels)]
    for i in idx:
        onehot[i] = 1
    return onehot

train_df = pd.read_csv('./titanic/train.csv')
train_df.head()

train_df.info()

print(train_df.dtypes)

train_df = pc.process_df(train_df)
train_df.head()

train_dataset = ds.Dataset(train_df)
len(train_dataset)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

input_sz = 6
hidden_sz = 3
out_sz = 2
net = model.Net(input_sz, hidden_sz, out_sz)
device = torch.device('cpu')
net = net.to(device)

learning_rate = 0.01
loss_func = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
epoch = 32

for e in range(epoch):
    for X, labels in train_dataloader:
        T = convert_label_to_onehot(labels)
        y = F.softmax(net(X.float()), dim=1)
        loss = loss_func(y, torch.FloatTensor(T))
        loss.backward()
        optimizer.step()

outdir = 'output'
dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
save_folder = os.path.join(outdir, dt)
if not os.path.exists(save_folder):
        os.makedirs(save_folder)

torch.save(net, os.path.join(save_folder, 'epoch_%02d' % (epoch)))





