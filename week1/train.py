from os import initgroups
import matplotlib.pyplot as plt
import torch
from torch._C import ConcreteModuleType
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from network import Net,CNN 
import time

# データセットをダウンロード(download=True)
dataroot = '学習データを保存するディレクトリのパス' 
train_data = MNIST(dataroot, train=True, download=True, transform=transforms.ToTensor())    ## 学習データを用意
train_loader = DataLoader(train_data,
                         batch_size=4,
                         shuffle=True)



#net = Net() # ネットワーク定義
net = CNN() # ネットワーク定義
criterion = nn.CrossEntropyLoss() # 損失関数(loss)を定義
optimizer = optim.SGD(net.parameters(), lr=0.01) # 最適化手法
writer = SummaryWriter()
images, labels = next(iter(train_loader))
writer.add_graph(net, images)
save_path = "重みファイルを保存するパス" 
# 学習スタート
start = time.time()
for epoch in range(3): # 何回繰り返すか(epoch数指定)
    running_loss = 0.0
    for i, data in enumerate(train_loader): # dataloaderで取り込んだデータセットを一つ一つネットワークに入力(学習)
        inputs, labels = data # 入力データと真値を格納
        optimizer.zero_grad () # 勾配情報をリセット
        outputs = net(inputs) # 順伝播

        loss = criterion(outputs, labels) # ロスの計算
        loss.backward() # 逆伝播
        optimizer.step() # パラメータの更新

        running_loss += loss.item() # lossの足し算
        torch.save(net.cpu().state_dict(), save_path) # 重みファイルの保存

        # 最後にlossの出力
        if i % 1000 == 0:
            if i == 0:
                continue
            print('epoch:%d iter:%d loss:%.3f' % (epoch + 1, i, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



