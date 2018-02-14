#author: Michael Shanahan 42839964
import tensorflow as tf
import numpy as np

#split the data and labels
def split_data(data, labels, proportion):
  #code from Canvas: written by Paul Quint, refactored by Brandon Geren
    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]]
#converts the given labels into white-hot vectors
def white_hot(labels):
  label_shape = ( len(labels), 10 ) #n rows, 10 columns
  new_labels = np.zeros(label_shape)
  idx = 0
  for i in labels:
    new_labels[int(idx),int(i)] = 1
    idx += 1
  return new_labels
#given a confusion matrix and a dimensionality, calculates the proportion of correct answers
def classification_rate(conf_matrix,dim):
  sum_entries = sum(sum(conf_matrix))
  sum_correct = 0
  for i in range(0,dim):
    sum_correct += conf_matrix[i,i]
  return float(sum_correct / sum_entries)