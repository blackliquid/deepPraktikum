
import tensorflow as tf
import numpy as np
import os
import scipy.misc


# output_img = graph.get_tensor_by_name("output_img:0")
# x = graph.get_tensor_by_name("x:0")
# batch_size = graph.get_tensor_by_name("batch_size:0")

batch_size_now = 2

values = np.load('celeba_variables.npz')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x")
batch_size = tf.placeholder(tf.int32, None, name="batch_size")


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
    return tf.nn.conv2d_transpose(x, W,output_shape, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# num_convs = 2
num_filters1 = 16
num_filters2 = 32
num_fc1 = 2048
num_fc2 = 512

# Rueckweg
channels = 48  # 8*3RGB also: MUSS DURCH 3 TEILBAR SEIN!

W_conv1 = tf.Variable(values['W_conv1_v'])
b_conv1 = tf.Variable(values['b_conv1_v'])

# x_flat = tf.reshape(x, [-1])
x_image = tf.reshape(x, [-1, 64, 64, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(values['W_conv2_v'])
b_conv2 = tf.Variable(values['b_conv2_v'])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # output last convlayer

W_fc1 = tf.Variable(values['W_fc1_v'])
b_fc1 = tf.Variable(values['b_fc1_v'])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*num_filters2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = tf.Variable(values['W_fc2_v'])
b_fc2 = tf.Variable(values['b_fc2_v'])

y_conv_ = tf.matmul(h_fc1, W_fc2)
y_conv = tf.reduce_mean(tf.add(y_conv_, b_fc2), axis=0, keep_dims=True)

W_fc1_r = tf.Variable(values['W_fc1_r_v'])
b_fc1_r = tf.Variable(values['b_fc1_r_v'])

h_fc1_r_ = tf.matmul(y_conv, W_fc1_r)
h_fc1_r = tf.add(h_fc1_r_, b_fc1_r)
h_fc1_r_flat = tf.reshape(h_fc1_r, [-1, 16, 16, channels])

W_conv2_r = tf.Variable(values['W_conv2_r_v'])
b_conv2_r = tf.Variable(values['b_conv2_r_v'])
output_shape_conv2r = [1, 32, 32, channels]

h_conv2_r = tf.nn.relu(deconv2d(h_fc1_r_flat, W_conv2_r, output_shape_conv2r) + b_conv2_r)  # deconvolution1

W_conv1_r = tf.Variable(values['W_conv1_r_v'])
b_conv1_r = tf.Variable(values['b_conv1_r_v'])
output_shape_conv1r = [1, 64, 64, channels]

h_conv1_r = deconv2d(h_conv2_r, W_conv1_r, output_shape_conv1r) + b_conv1_r  # deconvolution 2
# output_img = tf.nn.softmax(tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1]),  name='output_img')
# output_img = tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1],  name='output_img')
output_img = tf.reshape(h_conv1_r, [1, 64, 64, 3, channels//3])
output_img = tf.reduce_mean(output_img, axis=4, name='output_img')

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

filelist = nums_to_filelist([1, 2])
batch = np.array([scipy.misc.imread('./Datasets/img_align_celeba_resized/'+bild) for bild in filelist])
img = sess.run(output_img, feed_dict={x: batch, batch_size: batch_size_now})

scipy.misc.imsave('imagesCeleba64x64_mean/image.png', img[0])
