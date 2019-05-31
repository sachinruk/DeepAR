import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticLSTM(nn.Module):
    def __init__(self, n_features, h, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, h, num_layers)
        self.linear = nn.Linear(h, n_features*2)

    def forward(self, input, h=None):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input, h)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        mu_sigma = self.linear(lstm_out)
        out_shape = mu_sigma.shape
        out_shape = out_shape[:-1] + (out_shape[-1]//2,2)
        mu_sigma = mu_sigma.view(out_shape)
        mu, sigma = mu_sigma[...,0], mu_sigma[...,1]
        sigma = torch.log(1 + torch.exp(sigma)) # make numbers positive
        return mu, sigma, self.hidden

    def sample(self, input, h, n_samples, t_ahead):
        """
        Returns one sample trajectory 
        """
        shape = input.shape
        y_sample = torch.zeros((t_ahead,)+shape[1:])
        for t in range(t_ahead):
            mu, sigma, h = self.forward(input, h)
            y_sample[t] = mu + sigma * torch.randn(shape)
            input = y_sample[t]

        return y_sample


class LSTM_(nn.Module):
    def __init__(self, f,h,n):
        super().__init__()
        self.lstm = nn.LSTM(f,h,n)
        self.linear = nn.Linear(h, f)
        self.h = None

    def forward(self, x, h=None):
        out, self.h = self.lstm(x, h)
        y = self.linear(out)

        return y, self.h
