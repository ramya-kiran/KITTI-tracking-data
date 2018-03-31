import tensorflow as tf 
import time
import numpy as np 
import scipy as sp 
from pointNet_model import *
import argparse

HEIGHT = 1000
WIDTH = 4
CHANNEL = 1
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
    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=800, num_threads=2, min_after_dequeue=200)

    points = tf.placeholder(tf.float32, [None, HEIGHT,WIDTH, CHANNEL], name='points')
    
