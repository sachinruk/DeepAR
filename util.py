from torch.nn.utils.rnn import *

def get_rnn_batch(batch_X, batch_Y):
    """
    Takes in a sequence of size T x n_features and returns a tensor of size 
    """
    len_X = [len(x) for x in batch_X]
    batch_X = pad_sequence(batch_X).transpose(0,1) # (B,L,D) -> (L,B,D)
    batch_Y = pad_sequence(batch_Y).transpose(0,1) # (B,L,D) -> (L,B,D)
    packedX = pack_padded_sequence(batch_X, len_X)
    packedY = pack_padded_sequence(batch_Y, len_X)
    
    return packedX, packedY

def create_X(X, T, bs):
    """
    Splits a dataset X of size N x F into batches of size bs x T x F.
    Outputs each batch as a generator.
    """
    dataY = X[1:].split(T)
    Ts = len(dataY)
    dataX = X.split(T)[:Ts]
    batches_per_epoch = len(dataX) // bs

    for i in range(bs):
        # batches will be spaced out by batches_per_epoch
        idx = range(i,len(dataX),batches_per_epoch) 
        yield get_rnn_batch([dataX[i] for i in idx], [dataY[i] for i in idx])

    