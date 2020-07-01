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
    
  num_train, num_dims = X.shape
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i] @ W # shape = (1, C)
    scores -= np.max(scores)

    correct_class = y[i]
    correct_class_score = scores[correct_class] # f_yi
    
    exp_score = 0.
    for cl in range(num_class):
        # exp(fj)
        exp_score += np.exp(scores[cl])
      
    # Gradient
    for cl in range(num_class):
        dW[:,cl] += (np.exp(scores[cl]) / exp_score) * X[i]

    # Li = -f_yi + log(sum(scores))    
    loss_i = -correct_class_score + np.log(exp_score)  
    
    loss += loss_i
    
    # grad: Xi * W_yi
    dW[:,correct_class] -= X[i]
    
  # Mean of loss
  data_loss = loss / num_train
  reg_loss = reg * np.sum(np.square(W))

  loss = data_loss + reg_loss
    
  dW = (dW / num_train) + (reg * 2 * W)
    
  # Gradient
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
  
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X @ W # shape: (N, C)
  scores -= np.max(scores)

  correct_scores = scores[np.arange(scores.shape[0]), y].reshape((num_train, 1)) # shape: (N, 1)
  
  # Log of sum of exp(score)
  exps_sum = np.sum(np.exp(scores), axis=1, keepdims=True) # shape: (N, 1)
  log_exps = np.log(exps_sum)

  losses = -correct_scores + log_exps
  
  data_loss = np.sum(losses) / num_train
  reg_loss = reg * np.sum(np.square(W))

  loss = data_loss + reg_loss

  # Derivative of L w.r.t W_yi = X[i]
    
  # Turn y in to shape (N, C) with 1
  y_bits = np.zeros((num_train, num_class))
  y_bits[np.arange(num_train), y] = 1
    
  dW -= X.T @ y_bits # (D, N) (N, C)

  # X_i * exp(s) / sum(exp(s_j)) 
  score_by_sum = np.exp(scores) / exps_sum # shape: (N, C)

  dW += X.T @ score_by_sum

  dW = (dW / num_train) + (reg * 2 * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

