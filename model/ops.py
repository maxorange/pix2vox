import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.get_variable('W', shape, initializer=tf.random_normal_initializer(0., 0.02))

def bias_variable(shape):
    return tf.get_variable('b', shape, initializer=tf.constant_initializer(0.))

def keep_prob(dropout, train):
    return tf.cond(train, lambda: tf.constant(dropout), lambda: tf.constant(1.))

def softmax_ce_with_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def sigmoid_ce_with_logits(logits, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

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

def batch_norm(x, train, name, decay=0.99, epsilon=1e-5):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [shape[-1]], initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
        pop_mean = tf.get_variable('pop_mean', [shape[-1]], initializer=tf.constant_initializer(0.), trainable=False)
        pop_var = tf.get_variable('pop_var', [shape[-1]], initializer=tf.constant_initializer(1.), trainable=False)

        if pop_mean not in tf.moving_average_variables():
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, pop_mean)
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, pop_var)

        def func1():
            # Execute at training time
            batch_mean, batch_var = tf.nn.moments(x, range(len(shape) - 1))
            update_mean = tf.assign_sub(pop_mean, (1 - decay)*(pop_mean - batch_mean))
            update_var = tf.assign_sub(pop_var, (1 - decay)*(pop_var - batch_var))
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

        def func2():
            # Execute at test time
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, epsilon)

        return tf.cond(train, func1, func2)
