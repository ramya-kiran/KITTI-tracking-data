import tensorflow as tf
import numpy as np

''' 
Ref : https://github.com/mchablani/deep-learning/blob/master/autoencoder/Convolutional_Autoencoder.ipynb '''

N = 200
WIDTH = 3
CHANNEL = 1 

def model(inputs):
    # Encoder Module
    training_phase= True
    batch_size = inputs.shape[0]
    mod_input = tf.expand_dims(inputs, -1)

    conv1 = conv_layer(mod_input, [1,3], 1, 7, "conv1", True, training_phase, 'VALID')

    conv2 = conv_layer(conv1, [1,1], 7, 14, "conv2", True, training_phase, 'VALID')
    conv3 = conv_layer(conv2, [1,1], 14, 28, "conv3", True, training_phase, 'VALID')

    pool1 = pool_operations(conv3, [1,N,1,1], [1,2,2,1], "pool1")

    # Decoder Module
    upsample1 = tf.image.resize_images(pool1, size=(5,3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv4 = conv_layer(upsample1, [1,3], 28, 28, "conv4", True, training_phase, 'SAME')
    upsample2 = tf.image.resize_images(conv4, size=(25,3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv5 = conv_layer(upsample2, [1,3], 28, 28, "conv5", True, training_phase, 'SAME')

    upsample3 = tf.image.resize_images(conv5, size=(50,3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv6 = conv_layer(upsample3, [1,3], 28, 28, "conv6", True, training_phase, 'SAME')

    upsample4 = tf.image.resize_images(conv6, size=(100,3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv7 = conv_layer(upsample4, [1,3], 28, 28, "conv7", True, training_phase, 'SAME')

    upsample5 = tf.image.resize_images(conv7, size=(200,3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv8 = conv_layer(upsample5, [1,3], 28, 28, "conv8", True, training_phase, 'SAME')
    
    conv9 = conv_layer(conv8, [1,3], 28, 14, "conv9",True, training_phase, 'SAME')
    conv10 = conv_layer(conv9, [1,3], 14, 7, "conv10",  True, training_phase, 'SAME')
    conv11 = conv_layer(conv10, [1,3], 7, 1, "conv11", True, training_phase, 'SAME')

    conv10 =tf.squeeze(conv11)
    return conv10, upsample1, upsample2



def conv_layer(in_image, fil_size, no_in, no_out, name, bn_norm, training_phase, pad_type):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [fil_size[0], fil_size[1], no_in, no_out], 
        	initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [no_out], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        conv = tf.nn.conv2d(in_image, weights, strides=[1,1,1,1], padding=pad_type)
        conv = tf.nn.bias_add(conv,biases)

        if bn_norm:
            conv = batch_norm(conv, conv.get_shape()[-1].value ,[0,1,2], training_phase, scope='bn')
        
        return tf.nn.relu(conv)

def pool_operations(in_val,ksize_value, stride_value, name):
    with  tf.variable_scope(name):
        pool_1 = tf.nn.max_pool(in_val, ksize=ksize_value, strides=stride_value, padding='VALID')
        return pool_1


def fc_layer(image, in_size, out_size, name, bn_norm, training_phase):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", [out_size], initializer=tf.random_normal_initializer(0.0))
        y = tf.nn.bias_add(tf.matmul(image, weights), biases)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)

        if bn_norm:
            conv = batch_norm(y, y.get_shape()[-1].value, [0,], 
                is_training=training_phase,scope='bn')

        return tf.nn.relu(y)


''' Ref: https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow '''

def batch_norm(x, n_out, moment_dims, phase_train, scope='bn'):

    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.cast(phase_train, tf.bool),mean_var_with_update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
