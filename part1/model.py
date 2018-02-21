#author: Michael Shanahan 42839964
import util
import tensorflow as tf
import numpy as np

regularizer = tf.contrib.layers.l2_regularizer(1.0)

def build_network(input_placeholder):
  conv_layer = convolutional_layer(input_placeholder)
  linear_layer = linear_layer(conv_layer)
  output_layer = tf.layers.dense(linear_layer, 7, name = 'output_layer')
  return output_layer

#architecture: convolutional -> pool2 -> convolutional -> pool2
#description: a simple architecture to build off of
def convolutional_layer(inputs):
  reshaped_inputs = tf.reshape(inputs, [-1, 129, 129, 1])

  filters = [32, 32]
  kernel_size = [3, 3]
  pool_size = [2, 2]
  strides = 2
  padding = 'same'

  with tf.name_scope("convolutional"):
    conv_1 = tf.layers.conv2d(
      reshaped_inputs,
      filters[0],
      kernel_size,
      padding = padding,
      activation = tf.nn.relu,
      kernel_regularizer = regularizer,
      bias_regularizer = regularizer,
      name = 'conv_1'
    )
    
    pool_1 = tf.layers.max_pooling2d(
      conv_1,
      pool_size,
      strides,
      padding = padding,
      name = 'pool_1'
    )
    
    conv_2 = tf.layers.conv2d(
      pool_1,
      filters[1],
      kernel_size,
      padding = padding,
      activation = tf.nn.relu,
      kernel_regularizer = regularizer,
      bias_regularizer = regularizer,
      name = 'conv_2'
    )
    
    pool_2 = tf.layers.max_pooling2d(
      conv_2,
      pool_size,
      strides,
      padding = padding,
      name = 'pool_2'
    )
    
    #this line is sort of confusing
    #ignore the -1, don't change it
    #the second value is important
    #the first two values (33 and 33) are the x and y dimensions of the image AFTER pooling
    #in this case it is 33, because we have 2 pooling layers
    #129 / 2 = 65 | 65 / 2 = 33 (ROUND UP)
    #the 32 comes from the number of filters in our final convolutional layer
    #just set it to whatever the number of filters in the final convolutional layer is
    flatten_conv = tf.reshape(pool_2,[-1,33 * 33 * 32])

  return flatten_conv

#architecture: dense -> dense
#description: a simple architecture to build off of
def linear_layer(inputs):
  layer_counts = [32, 32]

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