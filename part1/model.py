import tensorflow as tf
import numpy as np
import os

def conv_block(inputs, filters, kernel_size, activation, dropout):
    # Conv & pooling layer 1
    with tf.name_scope('Conv_Pool_1') as scope:
        conv1 = tf.layers.conv2d(inputs=inputs,
                         filters=filters[0],
                         kernel_size=kernel_size[0],
                         padding="same",
                         activation=activation,
                         name='conv_1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                               pool_size=[2, 2],
                               strides=2,
                               name='pool_1')
        
    # Conv & pooling layer 2    
    with tf.name_scope('Conv_Pool_2') as scope:
        conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=filters[1],
                         kernel_size=kernel_size[1],
                         padding="same",
                         activation=activation,
                         name='conv_2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                               pool_size=[2, 2],
                               strides=2,
                               name='pool_2')
                               
    # Conv & pooling layer 3
    #with tf.name_scope('Conv_Pool_3') as scope:
    #    conv3 = tf.layers.conv2d(inputs=pool2,
    #                    filters=filters[2],
    #                    kernel_size=kernel_size[2],
    #                    padding="same",
    #                    activation=activation,
    #                    name='conv_3')
    #    pool3 = tf.layers.max_pooling2d(inputs=conv3,
    #                            pool_size=[2, 2],
    #                            strides=2,
    #                            name='pool_3')

    # Logit layer
    flat = tf.reshape(pool2,[-1, 32*32*64])
    #flat = tf.reshape(pool3,[-1, 16*16*128])
    #flat = tf.layers.dense(flat,1024)
    #flat = tf.layers.dense(flat, 32768, name='dense_1')
    flat = tf.layers.dense(flat, 30198, name='dense_1')
    dropout = tf.layers.dropout(inputs=flat,
                           rate=dropout, name='dropout')
    logits = tf.layers.dense(dropout, units=7, name='logits')
        
    return logits
