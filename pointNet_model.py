''' 
This file contains functions to create the PointNet model used for the task of classification 
Reference : "https://github.com/charlesq34/pointnet"
'''

import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 

def model(inputs, batch_size, training_phase):
	# BxNx3x1 --> BxNx1x64
	conv1 = conv_layer(inputs, [1,3], 1, 64, "conv1", bn_norm = True, training_phase)
	# BxNx1x64 --> BxNx1x64
	conv2 = conv_layer(conv1, [1,1], 64, 64, "conv2", bn_norm = True, training_phase)
	# BxNx1x64 --> BxNx1x64
	conv3 = conv_layer(conv2, [1,1], 64, 64, "conv3", bn_norm = True, training_phase)
	# BxNx1x64 --> BxNx1x128
	conv4 = conv_layer(conv3, [1,1], 64, 128, "conv4", bn_norm = True, training_phase)
	# BxNx1x64 --> BxNx1x1024
	conv5 = conv_layer(conv4, [1,1], 128, 1024, "conv5", bn_norm = True, training_phase)
	# Max pooling layer BxNx1x1024 --> Bx1x1x1024
	pool1 = conv_layer(conv5, [1,batch_size,1,1], [1,2,2,1], "pool1")
	pool1 = tf.reshape(pool1, [batch_size, -1])
	fc_1 = fc_layer(pool1,1024, 512, "fc_1", bn_norm = True, training_phase)
	fc_2 = fc_layer(fc_1,512, 256 ,"fc_1", bn_norm = True, training_phase)
	fc_2 = tf.nn.dropout(fc2, keep_prob=0.5, noise_shape=None)
	fc_3 = fc_layer(fc_2,256, 40, "fc_1", bn_norm = True, training_phase)

	return fc_3


def conv_layer(in_image, fil_size, no_in, no_out, name, bn_norm, training_phase):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [fil_size[0], fil_size[1], no_in, no_out], 
        	initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [no_out], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        conv = tf.nn.conv2d(in_image, weights, strides=[1,1,1,1], padding='SAME')
        conv = tf.nn.bias_add(conv,biases)

        if bn_norm:
			conv = batch_norm(conv, conv.get_shape()[-1].value, [0,1,2] 
				is_training=training_phase,scope='bn')
        
        return tf.nn.relu(conv)

def pool_operations(in_val,ksize_value, stride_value, name):
    with  tf.variable_scope(given_name):
        pool_1 = tf.nn.max_pool(in_val, ksize=ksize_value, strides=stride_value, padding='VALID')
        return pool_1


def fc_layer(image, in_size, out_size, name, bn_norm, training_phase):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [out_size], initializer=tf.random_normal_initializer(0.0))
        y = tf.ann.bias_add(tf.matmul(image, weights), biases)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)

		if bn_norm:
			conv = batch_norm(y, y.get_shape()[-1].value, [0,] 
				is_training=training_phase,scope='bn')

        return tf.nn.relu(y)

''' Ref: https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow '''

def batch_norm(x, n_out, moment_dims, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

