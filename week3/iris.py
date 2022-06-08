"""
今回はこちらのコードを実装してください
ライブラリが足りない場合は各自インストールしてください
この課題では穴埋めなどはありませんので、自分の力だけでコードが動くようにしてみてください
参考サイトはREADMEに載せているのでコードをゼロから実装してもかまいません
前回実装したMNISTと処理過程で似ている部分などを分析してください
次回はYOLOと言われる深層学習モデルを自分で実装してもらいます
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
from net import Net

## データのロード
iris = load_iris()
data = iris.data
label = iris.target

## データを分割
train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.2)
print("train_data size: {}".format(len(train_data)))
print("test_data size: {}".format(len(test_data)))
print("train_label size: {}".format(len(train_label)))
print("test_label size: {}".format(len(test_label)))

## データをTensor型に変換
train_x = torch.Tensor(train_data)
test_x = torch.Tensor(test_data)
train_y = torch.LongTensor(train_label)
test_y = torch.LongTensor(test_label)

## データをテンソルのデータセットする
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

## データセットの際の細かい設定をする
train_batch = DataLoader(
    dataset = train_dataset,#データセットの指定
    batch_size = 5,#バッチサイズの指定
    shuffle = True,#シャッフルするかどうかの指定
    num_workers = 2#コア数
)
test_batch = DataLoader(
    dataset = test_dataset,
    batch_size = 5,
    shuffle = False,
    num_workers = 2
)

## for文で回してサイズを確認する
for data, label in train_batch:
    print("batch data size: {}".format(data.size()))
    print("batch label size: {}".format(label.size()))
    break

## データセットの際の細かい設定をする
train_batch = DataLoader(
    dataset = train_dataset,#データセットの指定
    batch_size = 5,#バッチサイズの指定
    shuffle = True,#シャッフルするかどうかの指定
    num_workers = 2#コア数
)
test_batch = DataLoader(
    dataset = test_dataset,
    batch_size = 5,
    shuffle = False,
    num_workers = 2
)

## for文で回してサイズを確認する
for data, label in train_batch:
    print("batch data size: {}".format(data.size()))
    print("batch label size: {}".format(label.size()))
    break

## パラメータ表示
#ハイパーパラメータ
D_in = 4#入力次元
H = 100#隠れ層次元
D_out = 3#出力次元
epoch = 100#学習回数

## デバイスの指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ネットワーク実行
net = Net(D_in, H, D_out).to(device)
print("Device: {}".format(device))

#損失関数の定義
criterion = nn.CrossEntropyLoss()

#最適化関数の定義
optimizer = optim.Adam(net.parameters())

## 学習に必要な空リストを作成
train_loss_list = []#学習損失
train_accuracy_list = []#学習データ正解率
test_loss_list = []#評価損失
test_accuracy_list = []#テストデータの正答率

#学習の実行
for I in range(epoch):
    #学習の進行状況を表示
    print('--------')
    print("Epoch: {}/{}".format(I + 1, epoch))
    #損失と正解率の初期化
    train_loss = 0#学習損失
    train_accuracy = 0#学習データの正答数
    test_loss = 0#評価損失
    test_accuracy = 0#テストデータの正答数
    #学習モードに設定
    net.train()
    #ミニバッチごとにデータをロードして学習
    for data, label in train_batch:
        data = data.to(device)
        label = label.to(device)
        #勾配を初期化
        optimizer.zero_grad()
        #データを入力して予測値を計算
        y_pred_prob = net(data)
        #損失を計算
        loss = criterion(y_pred_prob, label)
        #勾配を計算
        loss.backward()
        #パラメータの更新
        optimizer.step()
        #ミニバッチごとの損失を蓄積
        train_loss += loss.item()
        #予測したラベルを予測確率から計算
        y_pred_label = torch.max(y_pred_prob, 1)[1]
        #ミニバッチごとに正解したラベル数をカウント
        train_accuracy += torch.sum(y_pred_label == label).item() / len(label)
    #ミニバッチの平均の損失と正解率を計算
    batch_train_loss = train_loss / len(train_batch)
    batch_train_accuracy = train_accuracy / len(train_batch)
    #評価モードに設定
    net.eval()
    #評価時に自動微分をゼロにする
    with torch.no_grad():
        for data, label in test_batch:
            data = data.to(device)
            label = label.to(device)
            #データを入力して予測値を計算
            y_pred_prob = net(data)
            #損失を計算
            loss = criterion(y_pred_prob, label)
            #ミニバッチごとの損失を備蓄
            test_loss += loss.item()
            #予測したラベルを予測確率から計算
            y_pred_label = torch.max(y_pred_prob, 1)[1]
            #ミニバッチごとに正解したラベル数をカウント
            test_accuracy += torch.sum(y_pred_label == label).item() / len(label)
    #ミニバッチの平均の損失と正解率を計算
    batch_test_loss = test_loss / len(test_batch)
    batch_test_accuracy = test_accuracy / len(test_batch)
    #エポックごとに損失と正解率を表示
    print("Train_Loss: {:.4f} Train_Accuracy: {:.4f}".format(batch_train_loss, batch_train_accuracy))
    print("Test_Loss: {:.4f} Test_Accuracy: {:.4f}".format(batch_test_loss, batch_test_accuracy))
    #損失と正解率をリスト化して保存
    train_loss_list.append(batch_train_loss)
    train_accuracy_list.append(batch_train_accuracy)
    test_loss_list.append(batch_test_loss)
    test_accuracy_list.append(batch_test_accuracy)

##結果の表示
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red', linestyle='--', label='Test_Loss')
plt.legend()

plt.figure()
plt.title('Train and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(1, epoch+1), train_accuracy_list, color='blue', linestyle='-', label='Test_Accuracy')
plt.plot(range(1, epoch+1), test_accuracy_list, color='red', linestyle='--', label='Test_Accuracy')
plt.legend()

plt.show()
