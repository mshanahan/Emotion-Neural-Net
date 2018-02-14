#author: Michael Shanahan 42839964
import util
import tensorflow as tf
import numpy as np

def my_model(layer_counts, inputs):
  initializer = tf.contrib.layers.xavier_initializer()
  regularizer = tf.contrib.layers.l2_regularizer(1.0)

  #layer_counts: a 4d array of integers
  #[0]: hidden 1 size, [1]: hidden 2 size
  divided_inputs = inputs / 255.0

  hidden_1 = tf.layers.dense(
    divided_inputs,
    layer_counts[0],
    activation = tf.nn.relu,
    bias_regularizer = regularizer,
    kernel_regularizer = regularizer,
    kernel_initializer = initializer,
    name = 'hidden_1')

  hidden_2 = tf.layers.dense(
    hidden_1,
    layer_counts[1],
    activation = tf.nn.relu,
    bias_regularizer = regularizer,
    kernel_regularizer = regularizer,
    kernel_initializer = initializer,
    name = 'hidden_2')
    
  hidden_3 = tf.layers.dense(
    hidden_2,
    layer_counts[2],
    activation = tf.nn.relu,
    bias_regularizer = regularizer,
    kernel_regularizer = regularizer,
    kernel_initializer = initializer,
    name = 'hidden_3')
    
  dropout_1 = tf.nn.dropout(
    hidden_3,
    0.90)
    
  output_layer = tf.layers.dense(
    dropout_1,
    10,
    kernel_initializer = initializer,
    name = 'output_layer')
    
  return output_layer