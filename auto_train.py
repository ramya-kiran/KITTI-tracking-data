import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 
from auto_model import *
import argparse


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='tf record filename')
    parser.add_argument('-b', '--batch-size', default=10, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=4000, type=int, help='num of epochs')
    parser.add_argument('-o', '--output', default='model', help='model output')
    parser.add_argument('-l', '--log', default='logs', help='log directory')
    args = parser.parse_args()

    filename_queue = tf.train.string_input_producer(args.train)
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([in_points], batch_size=args.batch_size, capacity=800, num_threads=2, min_after_dequeue=200)

    points = tf.placeholder(tf.float32, [None, N,WIDTH, CHANNEL], name='points')

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=points, logits=decoder_out)

    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.001).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(args.epochs):
            p= sess.run(batch)
            loss_val, _ = sess.run([cost, opt], feed_dict={points: p})
            
            # Print training accuracy every 100 epochs
            if (i+1) % 100 == 0:
                print('loss val {}: {:.2f}'.format(i+1, sess.run(loss_val, feed_dict={points: p})))
                
            if (i+1) % 1000 == 0:
                params = saver.save(sess, '{}_{}.ckpt'.format(args.output, i+1))
                print('Model saved: {}'.format(params))
                
        coord.request_stop()
        coord.join(threads)
        

    
