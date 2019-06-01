from torch.nn.utils.rnn import *

def get_rnn_batch(batch_X, batch_Y):
    """ Takes a batch of data and packs them in the format required for RNNs.

    Parameters
    ----------
    batch_X: Array of torch Tensors
    batch_Y: Array of torch Tensors

    Returns
    -------
    packedX: tuple of torchTensors and batch lengths
        Packed inputs for RNNs discarding padding.
    packedY: tuple of torchTensors and batch lengths
        Packed inputs for RNNs discarding padding.
    """
    len_X = [len(x) for x in batch_X]
    batch_X = pad_sequence(batch_X) 
    batch_Y = pad_sequence(batch_Y)
    packedX = pack_padded_sequence(batch_X, len_X)
    packedY = pack_padded_sequence(batch_Y, len_X)
    
    return packedX, packedY

def create_X(X, T, bs):
    """Creates a generator for sequence to sequence models. 
    Batches are generated for stateful training.

    Parameters
    ----------
    X: pytorch Tensor
        A N x F tensor where N is the number of time steps and 
        F is the number of features
    T: int
        Maximum length of sequence that X will break up into.
    bs: int
        Batch size of dataset

    Returns
    -------
    packedX, packedY: generator of torch Tensors
    """
    # Split data into datasets of size T steps
    dataY = X[1:].split(T)
    dataX = X[:-1].split(T)

    batches_per_epoch = len(dataX) // bs
    for i in range(bs):
        # batches will be spaced out by batches_per_epoch
        idx = range(i,len(dataX),batches_per_epoch) 
        packedX, packedY = get_rnn_batch([dataX[i] for i in idx], [dataY[i] for i in idx])
        yield packedX, packedY
