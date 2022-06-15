import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_sz, hidden_sz, out_sz):
        super(Net, self).__init__()
        self.f1 = nn.Linear(input_sz, hidden_sz)
        self.f2 = nn.Linear(hidden_sz, out_sz)
        
    def forward(self, x):
        h1 = torch.sigmoid(self.f1(x))
        y = self.f2(h1)
        
        return y