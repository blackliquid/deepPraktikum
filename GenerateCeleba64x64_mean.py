
import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc


sess = tf.Session()
saver = tf.train.import_meta_graph('modelCeleba64x64.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
output_img = graph.get_tensor_by_name("output_img:0")
x = graph.get_tensor_by_name("x:0")
batch_size = graph.get_tensor_by_name("batch_size:0")

batch_size_now = 50


import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x")
batch_size = tf.placeholder(tf.int32, None,  name="batch_size")

#def cut_file(file):   #FILE HAS TO BE OF SHAPE 178x218
    #file = file[[21:-21],[1:-1]]

def random_filelist(batch_size):
    index = np.random.uniform(1, 202599.99, batch_size)
    index = index.astype(int)
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist

def cut_filelist(list):
    list = list[:, 21:-21, 1:-1, :]
    #    for i in range(len(filelist)):
    #        filelist[i] = filelist[i, 21:-21, 1:-1]
    return list

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W,output_shape, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


batch_size_now = 2

#num_convs = 2
num_filters1 = 16
num_filters2 = 32
num_fc1 = 2048
num_fc2 = 512

#Rueckweg
channels = 48   #8*3RGB also: MUSS DURCH 3 TEILBAR SEIN!

W_conv1 = weight_variable([5, 5, 3, num_filters1])
b_conv1 = bias_variable([num_filters1])

#x_flat = tf.reshape(x, [-1])
x_image = tf.reshape(x, [-1,64,64,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, num_filters1, num_filters2])
b_conv2 = bias_variable([num_filters2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #output last convlayer

W_fc1 = weight_variable([16 * 16 * num_filters2, num_fc1])
b_fc1 = bias_variable([num_fc1])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*num_filters2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([num_fc1, num_fc2])
b_fc2 = bias_variable([num_fc2])

y_conv_ = tf.matmul(h_fc1, W_fc2)
y_conv = tf.add(y_conv_, b_fc2)



W_fc1_r = weight_variable([num_fc2, 16*16*channels])
b_fc1_r = bias_variable([16*16*channels])

h_fc1_r_ = tf.matmul(y_conv, W_fc1_r)
h_fc1_r = tf.add(h_fc1_r_, b_fc1_r)
h_fc1_r_flat = tf.reshape(h_fc1_r, [-1, 16, 16, channels])

W_conv2_r = weight_variable([2, 2, channels, channels])
b_conv2_r = bias_variable([channels])
output_shape_conv2r = [batch_size, 32, 32, channels]

h_conv2_r = tf.nn.relu(deconv2d(h_fc1_r_flat, W_conv2_r, output_shape_conv2r) + b_conv2_r) #deconvolution1

W_conv1_r = weight_variable([2, 2, channels, channels])
b_conv1_r = bias_variable([channels])
output_shape_conv1r = [batch_size, 64, 64, channels]

h_conv1_r = deconv2d(h_conv2_r, W_conv1_r, output_shape_conv1r)+ b_conv1_r #deconvolution 2
#output_img = tf.nn.softmax(tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1]),  name='output_img')
#output_img = tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1],  name='output_img')
output_img = tf.reshape(h_conv1_r,[batch_size,64,64,3,channels//3])
output_img = tf.reduce_mean(output_img, axis=4, name = 'output_img')

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(x_flat * tf.log(output_img), reduction_indices=[0]))#manual cross-entropy. Numerically instable
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_flat, logits=output_img, dim=0)) #nice entropy function. Doesnt work
#least_squares = tf.reduce_sum(tf.multiply(x_flat- output_img,x_flat- output_img))
least_squares = tf.reduce_sum(tf.multiply(x - output_img,x - output_img))
train_step = tf.train.AdamOptimizer(1e-4).minimize(least_squares)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


for i in range(10000):
    filelist = random_filelist(batch_size_now)
    batch = np.array([scipy.misc.imread('./img_align_celeba_resized/'+bild) for bild in filelist])
    #batch = cut_filelist(batch)
    [_, loss_val] = sess.run([train_step, least_squares], feed_dict={x: batch, batch_size: batch_size_now})
    if i%100 == 0:
        print("step %d, loss %g" %(i, loss_val))

save_path = saver.save(sess, os.path.join(os.getcwd(), 'modelCeleba64x64'))
#save_path = saver.save(sess, 'model')
print("Model saved in file: %s" % save_path)