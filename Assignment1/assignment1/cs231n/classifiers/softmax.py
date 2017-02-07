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
  num_trains = X.shape[0]
  num_features = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_trains):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores)
    correct_exp_scores = exp_scores[y[i]]     
    loss += -np.log(correct_exp_scores/sum_exp_scores)
    for j in xrange(num_features):
      score_per = exp_scores[j] / sum_exp_scores
      if (j == y[i]):
        dW[:,j] += (score_per - 1) * X[i].T
      else :
        dW[:,j] += score_per * X[i].T     
    
  loss /= num_trains
  dW /= num_trains
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_trains = X.shape[0]
  num_features = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_pred = X.dot(W)
  exp_pred = np.exp(y_pred)
  sum_exp_pred = np.sum(exp_pred, axis=1)
  pred_per = exp_pred / sum_exp_pred[:,None]
  loss = -np.sum(np.log(pred_per[range(num_trains),y]))
  offset = np.zeros_like(y_pred)
  offset[range(num_trains),y] = 1;
  dW = X.T.dot(pred_per - offset)
  
  loss /= num_trains
  dW /= num_trains
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

