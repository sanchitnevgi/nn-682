import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # Computation graph
  # f = WX
  # g = f - s_y + 1
  # h = max(g, 0)
    
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W) # f, shape: (C,)
    correct_class_score = scores[y[i]]
    correct_class = y[i] 

    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1

      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,correct_class] -= X[i]

  dW = (dW / num_train) + (2 * reg * W)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # shape: (N, C)
  true_label_scores = scores[np.arange(scores.shape[0]), y].reshape((num_train, 1))
   
  margins = scores - true_label_scores + 1 # shape: (N, C)
  
  # Set margins of correct_class to zero
  margins[np.arange(margins.shape[0]), y] = 0
  
  # max(0, sj - sy + 1)
  margins = np.maximum(margins, 0)

  total_margin = margins.sum()
  
  loss = (total_margin / num_train) + (reg * np.sum(np.square(W)))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins_mask = margins > 0 # shape: (N, C)

  X_sums = X.T @ margins_mask # shape: (D, C)

  dW += X_sums
  
  # Take y and turn into 1, 0s of shape (N,C)
  y_bits = np.zeros((num_train, num_classes))
  y_bits[np.arange(y_bits.shape[0]), y] = 1
  y_bits = X.T @ y_bits 

  dW -= y_bits
  
  dW = (dW / num_train) + (2 * reg * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
