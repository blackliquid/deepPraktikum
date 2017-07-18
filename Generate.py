from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc

sess = tf.Session()
saver = tf.train.import_meta_graph('model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
output_img = graph.get_tensor_by_name("output_img:0")
x = graph.get_tensor_by_name("x:0")
batch_size = graph.get_tensor_by_name("batch_size:0")

batch_size_now = 50
batch = mnist.train.next_batch(50)[0]
result = sess.run(output_img, feed_dict = {x: batch, batch_size:50})
img = np.reshape(result, [50, 28, 28])


for i in range(0, batch_size_now):
    orig_img = np.reshape(batch[i], [28, 28])
    scipy.misc.imsave('./images/orig_%s.png'%i, orig_img)
    generated_img = img[i]
    scipy.misc.imsave('./images/generated_%s.png'%i, generated_img)

   # scipy.misc.imshow(orig_img)
   # scipy.misc.imshow(generated_img)


