import tensorflow as tf
from DiscriminatorNet import DiscriminatorNet
import numpy as np
import scipy.misc

def batchNorm(batch, eps = 0.01, gamma = 1, beta = 0):
    batch_size = batch.shape[0]
    xdim = batch.shape[1]
    ydim = batch.shape[2]
    channels = batch.shape[3]

    avg = np.average(batch, 0)
    stddev = np.std(batch, 0)
    stddev *= stddev #we want sigma² here

    #array for normalised batch

    batch_norm=[]

    #array for the scaled and shifted batch

    batch_scaled = []

    #calculate the normalised and shifted/scaled batch

    for i in range(0, batch_size):
        batch[i] = avg-batch[i] #subtract from avg
        batch[i]/= np.sqrt(stddev+eps) #normalize
        batch[i] = gamma*batch[i]+beta #scale and shift

def readBatch(batch_size, max_range):
    for i in range(0, batch_size):
        rand = np.random.randint(1, max_range, size=batch_size)
        batch = np.ndarray(shape=[batch_size, 64, 64, 3])
        for i in rand:
            img = scipy.misc.imread("../Datasets/img_align_celeba_resized/%06d.png" % i)
            batch[0] = img
    return batch, rand

def split_lines(lines):
    if isinstance(lines, int):
        lines = [lines]
    result = []
    for line in lines:
        split_line = line.split(' ')
        split_line = [i for i in split_line if i != '']
        split_line[0] = split_line[0][0:split_line[0].index('.')]
        result.append([int(i) for i in split_line])
    return result

def split_lines3D(lines):
    # Important: Do not change the list entries before converting them to an np.array!!!
    if isinstance(lines, int):
        lines = [lines]
    result = []
    for line in lines:
        split_line = line.split(' ')
        result_line = []
        liste_1 = [1, 0]
        liste_m1 = [0, 1]
        for i in split_line:
            if i == '1':
                result_line.append(liste_1)
            elif i == '-1':
                result_line.append(liste_m1)
        result.append(result_line)
    return result

def read_lines(line_nums):
    line_nums = [i+2 for i in line_nums]
    n = len(line_nums)
    max_num = max(line_nums)
    dict_line_nums = {line_nums[i]:[] for i in range(n)}
    for i in range(n):
        dict_line_nums[line_nums[i]].append(i)
    result = n*[0]
    with open('../Datasets/Anno/list_attr_celeba.txt') as file:
        for num, line in enumerate(file):
            if num in dict_line_nums:
                for i in dict_line_nums[num]:
                    result[i] = line.strip()
            elif num > max_num:
                break
    return result

def newBatch():
    batch, rand = readBatch(batch_size, max_range)
    print(rand)
    batchNorm(batch)
    read = read_lines(rand)
    attribs = np.array(split_lines3D(read))
    labels = attribs[:, 3, :]
    return batch, labels

sdev = 0.1
numIter = 100000
numPics = 202599
batch_size = 64
max_range = 1000

discriminatorNet = DiscriminatorNet(sdev)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


for i in range(numIter):
    if i % 100 is 0:
        batch, labels = newBatch()
        train_accuracy = discriminatorNet.accuracy.eval(feed_dict={discriminatorNet.input_batch: batch, discriminatorNet.labels:labels, discriminatorNet.batch_size:batch_size}, session = sess)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    discriminatorNet.optimizer.run(feed_dict = {discriminatorNet.input_batch: batch, discriminatorNet.labels:labels, discriminatorNet.batch_size:batch_size}, session = sess)

save_path = saver.save(sess, 'model/discriminator_attract_100kIter')
print("Model saved in file: %s" % save_path)