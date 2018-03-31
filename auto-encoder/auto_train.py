import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 
from auto_model import *
import argparse
from read_point_data import *


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='list of image(s)')
    parser.add_argument('-b', '--batch-size', default=10, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=4000, type=int, help='num of epochs')
    parser.add_argument('-o', '--output', default='model', help='model output')
    parser.add_argument('-l', '--log', default='logs', help='log directory')
    args = parser.parse_args()

    # filename_queue = tf.train.string_input_producer(args.train)
    p = read_points(args.source)
    # print(p.shape)
    batch = tf.random_shuffle(p)

    points = tf.placeholder(tf.float32, [None, N,WIDTH], name='points')

    decoder_out, up_1, up_2 = model(points)

    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=points, logits=decoder_out)
    loss = tf.losses.mean_squared_error(points, decoder_out, weights=1.0, scope=None, 
                                        loss_collection=tf.GraphKeys.LOSSES, 
                                        reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

   # cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(args.epochs):
            # print(p.shape)
            p = sess.run(batch)
            loss_val, _ = sess.run([cost, opt], feed_dict={points: p})
            
            #Print training accuracy every 100 epochs
            if (i+1) % 10 == 0:
                print('loss val {}: {}'.format(i+1, loss_val))
               # print(sess.run(up_2,feed_dict={points: p}))
                
            if (i+1) % 113 == 0:
                print('loss val {}: {}'.format(i+1, loss_val))
                params = saver.save(sess, '{}_{}.ckpt'.format(args.output, i+1))
                print('Model saved: {}'.format(params))
                
        coord.request_stop()
        coord.join(threads)
        

    
