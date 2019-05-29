import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticLSTM(nn.Module):
    def __init__(self, n_features, h, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, h, num_layers)
        self.linear = nn.Linear(h, n_features*2)
        # self.h = h
        # self.n = n_features

    # def init_hidden(self):
    #     # This is what we'll initialise our hidden state as
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        mu_sigma = self.linear(lstm_out)
        out_shape = mu_sigma.shape
        out_shape = out_shape[:-1] + (out_shape[-1]//2,2)
        mu_sigma = mu_sigma.view(out_shape)
        mu, sigma = mu_sigma[...,0], mu_sigma[...,1]
        sigma = torch.log(1 + torch.exp(sigma)) # make numbers positive
        return mu, sigma, self.hidden

    def sample(self, input, n_samples, t_ahead):
        # pass
        # # for 
        mu, sigma, h = self.forward(input)
        

