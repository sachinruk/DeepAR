import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from util import get_rnn_batch

from model import ProbabilisticLSTM

x = torch.arange(0,30,0.05)
y = [torch.sin(x), torch.sin(x-np.pi), torch.sin(x-np.pi/2)]
y = torch.stack(y)
y = y.t()

for y_ in y:
    plt.plot(x.cpu().numpy(), y_.cpu().numpy())
plt.show()

train_len = int(len(x) * 0.8)
train_y, test_y = y[:train_len], y[train_len:]

bs = 30
batches = get_rnn_batch(train_y, bs)

for x,y in batches:
    


