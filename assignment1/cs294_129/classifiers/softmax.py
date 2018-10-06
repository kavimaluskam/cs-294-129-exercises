import numpy as np
from random import shuffle

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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in xrange(num_train):
    scores = np.dot(X[i, :], W)
    correct_class_score = scores[y[i]]
    denominator = np.sum(np.exp(scores))
    loss += -1 * correct_class_score
    loss += np.log(denominator)
    
    for j in xrange(num_classes):
        dW[:, j] += X[i, :] * 1/denominator * np.exp(scores[j])
    dW[:, y[i]] -= X[i, :]

  # # Right now the loss is a sum over all training examples, but we want it
  # # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  import math
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  data_loss = 0.0
  full_loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  scores = np.dot(X, W)
  # === more detailed steps for reference, but may experience overflow ====
  # correct_class_score = scores[range(num_train), y]
  # full_loss_pt1 = -1 * np.sum(correct_class_score)
  # full_loss_pt2 = np.sum(np.log(np.sum(np.exp(scores), axis=1)))
  # full_loss = full_loss_pt1 + full_loss_pt2
  # data_loss = np.sum(full_loss) / num_train
  # reg_loss = 0.5 * reg * np.sum(W * W)
  # loss = data_loss + reg_loss
  # ======
 
  # instability will never make you down
  exp_scores = np.exp(scores + np.argmax(scores, axis=1)[:, np.newaxis])
  # exp_scores = np.exp(scores)
  full_loss = -1 * np.log(exp_scores[range(num_train), y]/np.sum(exp_scores, axis=1))
  data_loss = np.sum(full_loss) / num_train
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss

  d_scores = np.zeros_like(scores)
  d_scores[range(num_train), y] = -1
  d_scores += exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis]
  dW = np.dot(X.T, d_scores) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
