import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc

sess = tf.Session()
saver = tf.train.import_meta_graph('model/discriminator_attract_100kIter.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

who = 'chris'
face = scipy.misc.imread('../Datasets/faces/%s.png' %who, mode='RGB')
face_tensor = np.reshape(face, [1, 64, 64, 3])

graph = tf.get_default_graph()
res_fc2 = graph.get_tensor_by_name("res_fc2:0")
input_batch = graph.get_tensor_by_name("input_batch:0")
batch_size = graph.get_tensor_by_name("batch_size:0")

result = sess.run(res_fc2, feed_dict = {input_batch: face_tensor})
print(result)
predicted_class = np.argmax(result)
if(predicted_class is 0):
    print('%s ist ATTRAKTIV' %who)
else:
    print('%s ist UNATTRAKTIV' %who )
prob = result[0][predicted_class]/(result[0][0]+result[0][1])
print('mit Wahrschinlichkeit %g' %prob)

