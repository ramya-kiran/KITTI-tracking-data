import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 

def model(inputs):
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

	return output


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