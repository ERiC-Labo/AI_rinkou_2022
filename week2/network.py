from torch._C import set_anomaly_enabled
import torch.nn as nn
import torch
import torch.nn.functional as F

# ニューラルネットワークを定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(□,□)    #全結合層に通す。（）内の引数は？
        self.l2 = nn.Linear(50,□)    #全結合層に通す。第二引数は？
        
    def forward(self, x):
        x = x.view(-1, 28 * 28) #データ形状を変更
        x = self.l1(x)  #全結合層をインスタンス化
        x = self.l2(x)  #全結合層をインスタンス化
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(7 * 7 * 32, 10)
      

    def forward(self, x):               
        h = self.layer1(x)          # torch.Size([4, 16, 14, 14])  
        h = self.layer2(h)          # torch.Size([4, 32, 7, 7])  
        y = self.relu(h)
        y = y.view(y.size(0), -1)    # torch.Size([4, 1568])
        y = self.fc(y)               # torch.Size([4, 10])

        return y


