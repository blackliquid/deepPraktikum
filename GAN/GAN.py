import tensorflow as tf
import numpy as np
from GeneratorNet import GeneratorNet
from DiscriminatorNet import DiscriminatorNet


def run(self):  # run the computation
    self.sess = tf.session()
    self.sess.run(tf.global_variables_initializer())

def save(self, path):  # save the graph
    saver = tf.train.Saver
    saver.save(self.sess, path)

def trainDiscriminator(self, discriminatorNet, numIter, data):

    for i in range(0, numIter):
        discriminatorNet.optimizer.run(feed_dict={})

def batchNorm(self, batch, gamma, beta, eps):
    batch_size = batch.shape[0]
    avg = np.average(batch, 0)
    stddev = np.std(batch, 0)
    stddev *= stddev #we want sigma² here

    #array for normalised batch

    batch_norm=[]

    #array for the scaled and shifted batch

    batch_scaled = []

    #calculate the normalised and shifted/scaled batch

    for i in batch_size:
        batch[i] = avg-batch[i]
        batch /= np.sqrt(stddev+eps)

    return batch

sdev = 0.1
batch_size = 50

discriminatorNet = DiscriminatorNet(sdev)
generatorNet = GeneratorNet(sdev, batch_size, discriminatorNet)
