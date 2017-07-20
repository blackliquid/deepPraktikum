import tensorflow as tf
from GeneratorNet import GeneratorNet
from DiscriminatorNet import DiscriminatorNet


def run(self):  # run the computation
    self.sess = tf.session()
    self.sess.run(tf.global_variables_initializer())

def save(self, path):  # save the graph
    saver = tf.train.Saver
    saver.save(self.sess, path)

sdev = 0.1
batch_size = 50

generatorNet = GeneratorNet(sdev, batch_size)


