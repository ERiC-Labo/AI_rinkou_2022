import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(Net, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    return x
