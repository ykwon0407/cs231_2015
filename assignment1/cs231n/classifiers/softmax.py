import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_trains=X.shape[1]
  num_classes=W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_trains):
    f = W.dot(X[:,i])
    f -= np.max(f)
    f.shape = (num_classes,1)

    numer = np.exp(f)
    denom = np.sum(numer)
    loss -= np.log(numer[y[i]]/denom)

    dW += (numer/denom)*(X[:,i].T)
    dW[y[i],:] -= X[:,i].T

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_trains
  dW /= num_trains
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_trains=X.shape[1]
  num_classes=W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = W.dot(X)
  f -= np.max(f, axis=0)

  numer = np.exp(f)
  denom = np.sum(numer, axis=0)
  ind = [y, xrange(num_trains)]
  loss = np.sum(-np.log(numer[ind]/denom))/num_trains

  dW += (numer/denom).dot(X.T)
  dW -= np.stack([np.sum(X.T[(y==i),:], axis=0) for i in xrange(num_classes)])

  dW /= num_trains
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW
