import tensorflow as tf

class DiscriminatorNet:

    def __init__(self, sdev):
        self.sdev = sdev

        self.defineWeights()
        self.defineGraph()
        self.defineLoss()

    def weight_variable(self, shape): #shortcuts for defining the filters
        initial = tf.truncated_normal(shape, stddev=self.sdev)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(self.sdev, shape=shape)
        return tf.Variable(initial)

    def defineWeights(self):
        self.input_batch = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="input_batch") #placeholder for input images, fake and real
        self.labels = tf.placeholder(tf.int64, shape=[None, 2], name="labels") #placeholder for labels : generated or real
        #self.batch_size = tf.placeholder(tf.int32, None, name="batch_size") #placeholder for batch_size

        #define weights for the convLayers 0-3
        '''conv 1 : 64*64*3 -> 64*64*96
        pool 1 : 64*64*96 -> 32*32*96
        conv 2 : 32*32*96 -> 32*32*192
        pool2 : 32*32*192 -> 16*16*192
        conv 3 : 16*16*192 -> 16*16*384
        pool 3 : 16*16*384 -> 8*8*384
        conv 4 : 8*8*384 -> 8*8*768
        pool 4 : 8*8*784 -> 4*4*784'''

        self.W_conv = []
        self.b_conv = []

        #add weight variables for the first convlayer

        self.W_conv.append(self.weight_variable([5, 5, 3, 32*3]))
        self.b_conv.append(self.bias_variable([32*3]))


        # number of channels for the last 3 layers

        conv_dims = [96, 192, 384]

        #add weight variables for last 3 convlayers

        for i in conv_dims:
            self.W_conv.append(self.weight_variable([5, 5, i, i * 2]))
            self.b_conv.append(self.bias_variable([i * 2]))

        #define weights for the two FC layers
        '''FC1 : 4*4*768 -> 1000
        FC2 : 1000 -> 2'''

        self.W_fc = []
        self.b_fc = []

        #define weights for FC1

        self.W_fc.append(self.weight_variable([4*4*768, 1000]))
        self.b_fc.append(self.bias_variable([1000]))

        #define weights for FC2

        self.W_fc.append(self.weight_variable([1000, 2]))
        self.b_fc.append(self.bias_variable([2]))


    def defineGraph(self):
        #define graph for convlayers
        #array for the results of the conv and pool layers

        self.res_conv = []
        self.res_pool = []

        #define first convlayer as result of the input

        self.res_conv.append(tf.nn.relu(tf.nn.conv2d(self.input_batch, self.W_conv[0], strides=[1, 1, 1, 1], padding='SAME')+self.b_conv[0]))
        #leaky ReLu?
        #self.res_conv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d(self.input_batch, self.W_conv[0], strides=[1, 1, 1, 1], padding='SAME')+self.b_conv[0]))

        self.res_pool.append(tf.nn.avg_pool(self.res_conv[0], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME' ))

        #define convlayer 1-3

        for i in range(1, 4):
            self.res_conv.append(tf.nn.relu(tf.nn.conv2d(self.res_pool[i-1], self.W_conv[i], strides=[1, 1, 1, 1], padding='SAME') + self.b_conv[i]))
            #leaky ReLu?
            #self.res_conv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d(self.res_pool[i-1], self.W_conv[i], strides=[1, 1, 1, 1], padding='SAME') + self.b_conv[i]))
            self.res_pool.append(tf.nn.avg_pool(self.res_conv[i], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME' ))

        self.res_conv3_flat = tf.reshape(self.res_pool[3], [-1, 4*4*768])


        #define array for FC layers

        self.res_fc = []

        #fill it with fc1 and fc2

        self.res_fc.append(tf.nn.relu(tf.matmul(self.res_conv3_flat, self.W_fc[0])+self.b_fc[0]))
        #leaky ReLu?
        #self.res_fc.append(tf.contrib.keras.layers.LeakyReLu(tf.matmul(self.res_conv3_flat, self.W_fc[0])+self.b_fc[0]))
        self.res_fc.append(tf.add(tf.matmul(self.res_fc[0], self.W_fc[1]),self.b_fc[1], name="res_fc2")) #no reLu here!


    def defineLoss(self):
        #define cross-entropy loss between labels and logits

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.res_fc[1]))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.res_fc[1], 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
