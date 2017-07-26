# from tensorflow.examples.tutorials.mnist import input_data
import random as rnd
import numpy as np
import scipy.misc

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
# sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
batch_size = tf.placeholder(tf.int32, None,  name="batch_size")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W,output_shape, strides=[1, 2, 2, 1], padding='VALID')

# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')

def image_input_data(batch_size):
    # data = rnd.normalvariate(0, sigma=0.1)
    # print(data)
    # return np.array(data, tf.float32, [batch_size, 10])
    return np.random.normal(0, 1000, [batch_size, 10])


# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
# x_flat = tf.reshape(x, [-1])
# x_image = tf.reshape(x, [-1,28,28,1])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2) #output last convlayer
#
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv_ = tf.matmul(h_fc1, W_fc2)

sess = tf.InteractiveSession()

sess2 = tf.Session()

batch_size_now = 5

saver = tf.train.import_meta_graph('model2.meta')
saver.restore(sess2, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_conv = tf.placeholder(tf.float32, [None, 10])

channels = 32

W_fc1_r = graph.get_tensor_by_name('W_fc1_r:0')
b_fc1_r = graph.get_tensor_by_name('b_fc1_r:0')

h_fc1_r_ = tf.matmul(y_conv, W_fc1_r)
h_fc1_r = tf.add(h_fc1_r_, b_fc1_r)
h_fc1_r_flat = tf.reshape(h_fc1_r, [-1, 7, 7, channels])

W_conv2_r = graph.get_tensor_by_name('W_conv2_r:0')
b_conv2_r = graph.get_tensor_by_name('b_conv2_r:0')
output_shape_conv2r = [batch_size, 14, 14, channels]

h_conv2_r = tf.nn.relu(deconv2d(h_fc1_r_flat, W_conv2_r, output_shape_conv2r) + b_conv2_r) # deconvolution1

W_conv1_r = graph.get_tensor_by_name('W_conv1_r:0')
b_conv1_r = graph.get_tensor_by_name('b_conv1_r:0')
output_shape_conv1r = [batch_size, 28, 28, channels]

h_conv1_r = deconv2d(h_conv2_r, W_conv1_r, output_shape_conv1r)+ b_conv1_r #deconvolution 2
# output_img = tf.nn.softmax(tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1]),  name='output_img')
output_img = tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1],  name='output_img')

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(x_flat * tf.log(output_img), reduction_indices=[0]))#manual cross-entropy. Numerically instable
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_flat, logits=output_img, dim=0)) #nice entropy function. Doesnt work
# least_squares = tf.reduce_sum((tf.multiply(x_flat- output_img),(x_flat- output_img)))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(least_squares)
#
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
#
#
# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     [_, loss_val] = sess.run([train_step, least_squares], feed_dict={x: batch[0], batch_size: 50})
#     if i%100 == 0:
#         print("step %d, loss %g" %(i, loss_val))
#
# save_path = saver.save(sess, 'model')
# print("Model saved in file: %s" % save_path)

sess.run(tf.global_variables_initializer())

input_data = image_input_data(batch_size_now)

result = sess.run(output_img, feed_dict = {y_conv: input_data, batch_size: batch_size_now})
img = np.reshape(result, [batch_size_now, 28, 28])


for i in range(0, batch_size_now):
    generated_img = img[i]
    scipy.misc.imsave('./images_generated/generated_%s.png'%i, generated_img)