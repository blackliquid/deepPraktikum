
import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc

def random_filelist(batch_size):
    index = np.random.uniform(1, 202599.99, batch_size)
    index = index.astype(int)
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist


sess = tf.Session()
saver = tf.train.import_meta_graph('modelCeleba64x64.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
output_img = graph.get_tensor_by_name("output_img:0")
x = graph.get_tensor_by_name("x:0")
batch_size = graph.get_tensor_by_name("batch_size:0")

batch_size_now = 50

filelist = random_filelist(batch_size_now)
batch = np.array([scipy.misc.imread('./img_align_celeba_resized/' + bild) for bild in filelist])
result = sess.run(output_img, feed_dict = {x: batch, batch_size:batch_size_now})
img = np.reshape(result, [50, 64, 64,3])

for i in range(0, batch_size_now):
    orig_img = np.reshape(batch[i], [64, 64, 3])
    scipy.misc.imsave('./imagesCeleba64x64_moreChannels/orig_%s.png'%i, orig_img)
    generated_img = img[i]
    scipy.misc.imsave('./imagesCeleba64x64_moreChannels/generated_%s.png'%i, generated_img)



