import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
import process as pc
import dataset2 as ds
import model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate week5')
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    args = parser.parse_args()
    return args

def test():
    test_X = torch.tensor(test_df.iloc[:,:].values)
    test_X2 = test_X.to(device)
    test_Y = net(test_X2.float())
    survived = torch.max(test_Y, dim=1)[1]
    test_paID = pd.read_csv('./titanic/gender_submission.csv')['PassengerId']
    sub_df = pd.DataFrame({"PassengerId":test_paID.values, "Survived":survived})
    print(sub_df)
    return sub_df

test_df = pd.read_csv('./titanic/test.csv')
test_df.head()

test_df = pc.process_df(test_df)
test_df.head()
test_dataset = ds.Dataset(test_df)

train_dataloader = DataLoader(test_dataset, shuffle=False, drop_last=True)

args = parse_args()
net = torch.load(args.network)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

sub_df = test()
sub_df.to_csv("./submission.csv", index=False)