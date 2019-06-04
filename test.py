import torch
import torch.nn as nn
from torch import optim

import numpy as np
from util import create_X
from model import LSTM_

import matplotlib.pyplot as plt

# x = torch.arange(1000)[:,None].repeat(1,3).float()
x = torch.arange(0,30,0.01)
y = [torch.sin(x), torch.sin(x-np.pi), torch.sin(x-np.pi/2)]
y = torch.stack(y)
y = y.t()
x = y

print('Shape', x.shape)


model = LSTM_(3, 5, 2)
opt = optim.Adam(model.parameters())
loss_func = nn.MSELoss()
losses = []
epochs = 30
t_back = 20

for epoch in range(epochs):
    batches = create_X(x, t_back, 5)
    for i, (batchX, batchY) in enumerate(batches):
        # print(i, batchX[0].shape, batchY[0].shape)
        o, h = model(batchX)
        # import pdb; pdb.set_trace()
        loss = loss_func(o, batchY[0])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss)
    model.init_hidden()

plt.plot(losses)
plt.show()

# x = torch.arange(0,30,0.01)
# y = [torch.sin(x), torch.sin(x-np.pi), torch.sin(x-np.pi/2)]
# y = torch.stack(y)
# y = y.t()
# x = y

# for y_ in y:
#     plt.plot(x.cpu().numpy(), y_.cpu().numpy())
# plt.show()

# train_len = int(len(x) * 0.8)
# train_y, test_y = y[:train_len], y[train_len:]

# bs = 30
# batches = get_rnn_batch(train_y, bs)

# for x,y in batches:
    


