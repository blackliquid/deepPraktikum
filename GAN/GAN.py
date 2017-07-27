import tensorflow as tf
import numpy as np
import scipy.misc
from GeneratorNet import GeneratorNet
from DiscriminatorNet import DiscriminatorNet


class GAN:

    def __init__(self):
        sdev = 0.1
        mean = 0.5
        numIter = 100000
        batch_size = 128
        numIter_disc = 1 #number of times the disc gets updated for every time the gen gets updated
        #max_range = 202599
        max_range = 1000

        #init placehodler

        self.definePlaceholder()

        #init nets

        generatorNet = GeneratorNet(mean, sdev, self.batch_size, self.random_numbers)
        discriminatorNet = DiscriminatorNet(mean, sdev, self.batch_size, generatorNet)

        #init loss

        self.defineLossAlt(generatorNet, discriminatorNet)

        #init sess

        self.createRunSess()

        #train the net

        self.train(numIter, batch_size, numIter_disc, generatorNet, max_range)

    def createRunSess(self):
        # run the computation

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def save(self, path):
        # save the graph

        saver = tf.train.Saver
        saver.save(self.sess, path)

    def definePlaceholder(self):
        #define the placehodler variables for the net

        #placeholder for input images, fake and real


        self.batch_size = tf.placeholder(tf.int32, shape = None,
                                          name="batch_size")

        #placeholder for batch_size

        self.input_batch = tf.placeholder(tf.float32, shape = [None, 64, 64, 3],
                                          name="input_batch")

        #placeholder for random numbers

        self.random_numbers = tf.placeholder(tf.float32, shape = [None, 100],
                                          name="random_numbers")

        self.labels = tf.placeholder(tf.float32, shape=None,
                                             name="labels")

    def generate_rand(self, batch_size):
        return np.random.uniform(-1., 1., [batch_size, 100])


    def train(self, numIter, batch_size, numIter_disc, generatorNet, max_range):
        print("starting training...")
        with self.sess.as_default():
            for i in range(1, numIter):
                for k in range(numIter_disc):
                    # train discriminator with batch of db images

                    batch_real, _ = self.newBatch(batch_size, max_range)
                    self.batchNorm(batch_real)

                    rand = self.generate_rand(batch_size)

                    _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss],
                                                   feed_dict={self.input_batch: batch_real, self.batch_size: batch_size, self.random_numbers: rand})
                    print("updating discriminator. Loss : %g " %D_loss_curr)

                if i % 500 is 0 :
                    print("saving image batch...")

                    # while training save some of the generated images
                    #save all of the batch_size generated images

                    rand = self.generate_rand(batch_size)
                    generated_img = self.sess.run(generatorNet.generated_img,
                                                  feed_dict={self.batch_size: batch_size, self.random_numbers: rand})

                    for j in range(batch_size):
                        scipy.misc.imsave("../Datasets/GAN_generated/iter_%d_no_%d.png" %(i, j), generated_img[j])

                else :
                    # train generator with batch_size generated images

                    rand = self.generate_rand(batch_size)
                    _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss],
                                                   feed_dict={self.batch_size:batch_size, self.random_numbers: rand})
                    print("updating generator. Iteration %d. Loss : %g" %(i, G_loss_curr))



    def batchNorm(self, batch, eps=0.01, gamma=1, beta=0):
        batch_size = batch.shape[0]
        xdim = batch.shape[1]
        ydim = batch.shape[2]
        channels = batch.shape[3]

        avg = np.average(batch, 0)
        stddev = np.std(batch, 0)
        stddev *= stddev  # we want sigma² here

        # array for normalised batch

        batch_norm = []

        # array for the scaled and shifted batch

        batch_scaled = []

        # calculate the normalised and shifted/scaled batch

        for i in range(0, batch_size):
            batch[i] = batch[i] -avg  # subtract from avg
            batch[i] /= np.sqrt(stddev + eps)  # normalize
            batch[i] = gamma * batch[i] + beta  # scale and shift

    def readBatch(self, batch_size, max_range):
        rand = np.random.randint(1, max_range, size=batch_size)
        batch = np.ndarray(shape=[batch_size, 64, 64, 3])
        for i in range(batch_size):
            img = scipy.misc.imread("../Datasets/img_align_celeba_resized/%06d.png" % rand[i])
            batch[i] = img
        return batch, rand

    def split_lines(self, lines):
        if isinstance(lines, int):
            lines = [lines]
        result = []
        for line in lines:
            split_line = line.split(' ')
            split_line = [i for i in split_line if i != '']
            split_line[0] = split_line[0][0:split_line[0].index('.')]
            result.append([int(i) for i in split_line])
        return result

    def split_lines3D(self, lines):
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

    def read_lines(self, line_nums):
        line_nums = [i + 2 for i in line_nums]
        n = len(line_nums)
        max_num = max(line_nums)
        dict_line_nums = {line_nums[i]: [] for i in range(n)}
        for i in range(n):
            dict_line_nums[line_nums[i]].append(i)
        result = n * [0]
        with open('../Datasets/Anno/list_attr_celeba.txt') as file:
            for num, line in enumerate(file):
                if num in dict_line_nums:
                    for i in dict_line_nums[num]:
                        result[i] = line.strip()
                elif num > max_num:
                    break
        return result
#

    def newBatch(self, batch_size, max_range):
        #read new random batch out of DB. max_range indicates that we can sample from image 0 to max_range.

        batch, rand = self.readBatch(batch_size, max_range)
        self.batchNorm(batch)
        read = self.read_lines(rand)
        attribs = np.array(self.split_lines(read))
        labels = attribs[:, 3]
        return batch, labels

    def defineLossReal(self, generatorNet, discriminatorNet):
        d_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d_scope')
        D_real, D_logit_real = discriminatorNet.defineGraph(self.input_batch)
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, logits = D_logit_real))
        self.D_solver = tf.train.AdamOptimizer().minimize(loss=self.D_loss_real, var_list=d_scope_vars)

    def trainReal(self, numIter, batch_size, numIter_disc, generatorNet, max_range):
        for i in range(numIter):
            batch, labels = self.newBatch(batch_size, max_range)
            self.batchNorm(batch)
            print(self.sess.run([self.D_loss_real, self.D_solver], feed_dict= {self.input_batch: batch , self.batch_size: batch_size, self.labels: labels})[0])


    def defineLoss(self, generatorNet, discriminatorNet):
        # define the graphs we need for the loss function

        G_sample = generatorNet.generated_img
        D_real, D_logit_real = discriminatorNet.defineGraph(self.input_batch)
        D_fake, D_logit_fake = discriminatorNet.defineGraph(G_sample)

        #get varaibles from different scopes. This is because we want to update only certain variables and not all in each training step

        d_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d_scope')
        g_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_scope')

        #define the loss function and minimizer for both nets

        self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        self.D_solver = tf.train.AdamOptimizer().minimize(loss = self.D_loss, var_list = d_scope_vars)
        self.G_solver = tf.train.AdamOptimizer().minimize(loss = self.G_loss, var_list = g_scope_vars)

    def defineLossAlt(self, generatorNet, discriminatorNet):
        G_sample = generatorNet.generated_img
        D_real, D_logit_real = discriminatorNet.defineGraph(self.input_batch)
        D_fake, D_logit_fake = discriminatorNet.defineGraph(G_sample)

        d_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='d_scope')
        g_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_scope')

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_logit_real), logits = D_logit_real))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(D_logit_fake), logits = D_logit_fake))

        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_logit_fake), logits = D_logit_fake))

        self.D_solver = tf.train.AdamOptimizer().minimize(loss=self.D_loss, var_list=d_scope_vars)
        self.G_solver = tf.train.AdamOptimizer().minimize(loss=self.G_loss, var_list=g_scope_vars)

gan = GAN()
