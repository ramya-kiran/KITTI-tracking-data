''' 
This file contains functions to create the PointNet model used for the task of classification 
Reference : "https://github.com/charlesq34/pointnet"
'''

import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 

def model(inputs, batch_size):
	# BxNx3x1 --> BxNx1x64
	conv1 = conv_layer(inputs, [1,3], 1, 64, "conv1", bn_norm = True)
	# BxNx1x64 --> BxNx1x64
	conv2 = conv_layer(conv1, [1,1], 64, 64, "conv2", bn_norm = True)
	# BxNx1x64 --> BxNx1x64
	conv3 = conv_layer(conv2, [1,1], 64, 64, "conv3", bn_norm = True)
	# BxNx1x64 --> BxNx1x128
	conv4 = conv_layer(conv3, [1,1], 64, 128, "conv4", bn_norm = True)
	# BxNx1x64 --> BxNx1x1024
	conv5 = conv_layer(conv4, [1,1], 128, 1024, "conv5", bn_norm = True)
	# Max pooling layer BxNx1x1024 --> Bx1x1x1024
	pool1 = conv_layer(conv5, [1,batch_size,1,1], [1,2,2,1], "pool1")
	pool1 = tf.reshape(pool1, [batch_size, -1])
	fc_1 = fc_layer(pool1,1024, 512, "fc_1", bn_norm = True)
	fc_2 = fc_layer(fc_1,512, 256 ,"fc_1", bn_norm = True)
	fc_2 = tf.nn.dropout(fc2, keep_prob=0.5, noise_shape=None)
	fc_3 = fc_layer(fc_2,256, 40, "fc_1", bn_norm = True)

	return fc_3


def conv_layer(in_image, fil_size, no_in, no_out, name, bn_norm):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [fil_size[0], fil_size[1], no_in, no_out], 
        	initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [no_out], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        conv = tf.nn.conv2d(in_image, weights, strides=[1,1,1,1], padding='SAME')

        if bn_norm:
			conv = tf.contrib.layers.batch_norm(conv, enter=True, 
				scale=True, is_training=phase,scope='bn')
        
        return tf.nn.relu(conv + biases)

def pool_operations(in_val,ksize_value, stride_value, name):
    with  tf.variable_scope(given_name):
        pool_1 = tf.nn.max_pool(in_val, ksize=ksize_value, strides=stride_value, padding='VALID')
        return pool_1


def fc_layer(image, in_size, out_size, name, bn_norm):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [out_size], initializer=tf.random_normal_initializer(0.0))
        y = tf.add(tf.matmul(image, weights), biases)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        
        if bn_norm:
            

        return y

