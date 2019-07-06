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

print('Shape of dataset:', x.shape)

loss_func = nn.MSELoss()

F, H, N = 3, 7, 2

losses = []
epochs = 50
t_back = 20
batch_size = 4

model = LSTM_(F, H, N, batch_size)
opt = optim.Adam(model.parameters())

for epoch in range(epochs):
    batches = create_X(x, t_back, batch_size)
    for i, (batchX, batchY, bs) in enumerate(batches):
        # print(i, batchX[0].shape, batchX[1])
        o, h = model(batchX, bs)
        loss = loss_func(o, batchY[0])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.data.cpu().numpy())
    model.init_hidden() # reset hidden state
    print(f'Epoch {epoch} with loss: {np.mean(losses[-20:])}')

plt.plot(losses)
plt.title('Losses')
plt.savefig('losses')
plt.show()

model.eval()
x_test = torch.arange(30,60,0.01)
y_test = [torch.sin(x_test), 
          torch.sin(x_test-np.pi), 
          torch.sin(x_test-np.pi/2)]
y_test = torch.stack(y_test)
y_test = y_test.t()
x_test = y_test

model.eval()
start = 10
y_pred = model.predict(x_test[:start].view(-1,1,F))
y_preds = [y_pred[-1:]]
N_ahead = 2000
for i in range(N_ahead):
    y_preds.append(model.predict(y_preds[-1]))
y_preds = torch.cat(y_preds).squeeze()

plt.plot(x_test[start:start+N_ahead+1].cpu().numpy(), c='b', label='true')
plt.plot(y_preds.detach().cpu().numpy(), c='r', label='pred')
plt.legend()
plt.savefig('predictions')
plt.show()
