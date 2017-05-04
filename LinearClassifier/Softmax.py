import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    dW_each = np.zeros_like(W)
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)    # N by C
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1, considering the Numeric Stability
    #prob = np.exp(f - f_max) / np.reshape(np.sum(np.exp(f - f_max), axis=1), (num_train, 1))    # N by C
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)    # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0
    for i in xrange(num_train):
        for j in xrange(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # D by C
    num_train, dim = X.shape

    f = X.dot(W)    # N by C
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1, considering the Numeric Stability
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)   # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0    # N by C
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W

    return loss, dW
