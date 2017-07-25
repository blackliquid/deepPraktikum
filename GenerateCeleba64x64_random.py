
import tensorflow as tf
import numpy as np
import numpy.random as rnd
import os
import scipy.misc


values = np.load('celeba_variables.npz')

batch_size_now = 50
sess = tf.InteractiveSession()


def random_filelist(batch_size):
    index = np.random.uniform(1, 202599.99, batch_size)
    index = index.astype(int)
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist

def nums_to_filelist(index):
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


num_fc2 = 512
# Rueckweg
channels = 48  # 8*3RGB also: MUSS DURCH 3 TEILBAR SEIN!

mu = values['mu']
sigma = values['sigma']

y_conv_v = np.reshape(np.array([rnd.multivariate_normal(mu, sigma) for i in range(batch_size_now)],
                               dtype=np.float32), [batch_size_now, num_fc2])
y_conv = tf.placeholder(tf.float32, [None, num_fc2])

W_fc1_r = tf.Variable(values['W_fc1_r_v'])
b_fc1_r = tf.Variable(values['b_fc1_r_v'])

h_fc1_r_ = tf.matmul(y_conv, W_fc1_r)
h_fc1_r = tf.add(h_fc1_r_, b_fc1_r)
h_fc1_r_flat = tf.reshape(h_fc1_r, [-1, 16, 16, channels])

W_conv2_r = tf.Variable(values['W_conv2_r_v'])
b_conv2_r = tf.Variable(values['b_conv2_r_v'])
output_shape_conv2r = [batch_size_now, 32, 32, channels]

h_conv2_r = tf.nn.relu(deconv2d(h_fc1_r_flat, W_conv2_r, output_shape_conv2r) + b_conv2_r)  # deconvolution1

W_conv1_r = tf.Variable(values['W_conv1_r_v'])
b_conv1_r = tf.Variable(values['b_conv1_r_v'])
output_shape_conv1r = [batch_size_now, 64, 64, channels]

h_conv1_r = deconv2d(h_conv2_r, W_conv1_r, output_shape_conv1r) + b_conv1_r  # deconvolution 2
# output_img = tf.nn.softmax(tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1]),  name='output_img')
# output_img = tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1],  name='output_img')
output_img = tf.reshape(h_conv1_r, [batch_size_now, 64, 64, 3, channels//3])
output_img = tf.reduce_mean(output_img, axis=4, name='output_img')

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

img = sess.run(output_img, feed_dict={y_conv: y_conv_v})

for i in range(batch_size_now):
    scipy.misc.imsave('imagesCeleba64x64_random/randomImage%06i.png' % (i+1), img[i])

