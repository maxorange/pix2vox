import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.get_variable('W', shape, initializer=tf.random_normal_initializer(0., 0.02))

def bias_variable(shape):
    return tf.get_variable('b', shape, initializer=tf.constant_initializer(0.))

def keep_prob(dropout, train):
    return tf.cond(train, lambda: tf.constant(dropout), lambda: tf.constant(1.))

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

def linear(x, shape, name, bias=False):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.matmul(x, W)
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def conv2d(x, shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def deconv2d(x, shape, output_shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-2]])
            h = h + b
        return h

def conv3d(x, shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-1]])
            h = h + b
        return h

def deconv3d(x, shape, output_shape, name, bias=False, stride=2):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')
        if bias:
            b = bias_variable([shape[-2]])
            h = h + b
        return h
