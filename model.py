#author: Michael Shanahan 42839964
import util
import tensorflow as tf
import numpy as np

regularizer = tf.contrib.layers.l2_regularizer(1.0)

def build_network(input_placeholder):
  my_conv_layer = convolutional_layer(input_placeholder)
  my_linear_layer = linear_layer(my_conv_layer)
  output_layer = tf.layers.dense(my_linear_layer, 7, name = 'output_layer')
  return output_layer

#architecture: pool2 -> convolutional -> pool2 -> pool2 -> convolutional -> dense -> dense -> output
#description: a layer that pools very aggressively and ends with a tiny image
def convolutional_layer(inputs):
  reshaped_inputs = tf.reshape(inputs, [-1, 129, 129, 1])

  filters = [32, 32]
  kernel_size = [3, 3]
  pool_size = [2, 2]
  strides = 2
  padding = 'same'

  with tf.name_scope("convolutional"):
    pool_1 = tf.layers.max_pooling2d(
      reshaped_inputs,
      pool_size,
      strides,
      padding = padding,
      name = 'pool_1'
    )

    conv_1 = tf.layers.conv2d(
      pool_1,
      filters[0],
      kernel_size,
      padding = padding,
      activation = tf.nn.relu,
      kernel_regularizer = regularizer,
      bias_regularizer = regularizer,
      name = 'conv_1'
    )
    
    pool_2 = tf.layers.max_pooling2d(
      conv_1,
      pool_size,
      strides,
      padding = padding,
      name = 'pool_2'
    )
    
    pool_3 = tf.layers.max_pooling2d(
      pool_2,
      pool_size,
      strides,
      padding = padding,
      name = 'pool_3'
    )
    
    conv_2 = tf.layers.conv2d(
      pool_3,
      filters[1],
      kernel_size,
      padding = padding,
      activation = tf.nn.relu,
      kernel_regularizer = regularizer,
      bias_regularizer = regularizer,
      name = 'conv_2'
    )
    
    flatten_conv = tf.reshape(conv_2,[-1,17 * 17 * 32])

  return flatten_conv

#architecture: dense -> dense
#description: a simple dense layer
def linear_layer(inputs):
  layer_counts = [8, 8]

  with tf.name_scope("linear"):
    hidden_1 = tf.layers.dense(
      inputs,
      layer_counts[0],
      activation = tf.nn.relu,
      bias_regularizer = regularizer,
      kernel_regularizer = regularizer,
      name = 'hidden_1')

    hidden_2 = tf.layers.dense(
      hidden_1,
      layer_counts[1],
      activation = tf.nn.relu,
      bias_regularizer = regularizer,
      kernel_regularizer = regularizer,
      name = 'hidden_2')

  return hidden_2