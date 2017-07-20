import tensorflow as tf

class DiscriminatorNet:

    def __init__(self):
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
        self.input_batch = tf.placeholder(tf.float32, shape=[None, 64 * 64, 3], name="input_batch") #placeholder for input images, fake and real
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels") #placeholder for labels : generated or real
        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size") #placeholder for batch_size

        #define weights for the convLayers 0-3
        '''conv 1 : 64*64*3 -> 32*32*96
        conv 2 : 32*32*96 -> 16*16*192
        conv 3 : 16*16*192 -> 8*8*384
        conv 4 : 8*8*384 -> 4*4*768'''

        self.W_conv = []
        self.b_conv = []
        conv_dims = [3, 96, 192, 384] # number of channels

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
        #array for the results of the convlayers

        self.res_conv = []

        #define first convlayer as result of the input

        self.res_conv.append(tf.nn.relu(tf.nn.conv2d(self.input_batch, self.W_conv[0])+self.b_conv[0]))

        #define convlayer 1-3

        for i in range(0, 4):
            self.res_conv.append(tf.nn.relu(tf.nn.conv2d(self.res_conv[i-1], self.W_conv[i]) + self.b_conv[i]))

        self.res_conv3_flat = tf.reshape(self.res_conv[3], [-1])

        #define array for FC layers

        self.res_fc = []

        #fill it with fc1 and fc2

        self.res_fc.append(tf.nn.relu(tf.nn.matmul(self.res_conv3_flat)+self.b_fc[0]))
        self.res_fc.append(tf.nn.add(tf.nn.matmul(self.res_fc[0]),self.b_fc[1], name="res_fc2"))


    def defineLoss(self):
        #define cross-entropy loss between labels and logits

        cross_entropy = tf.nn.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.res_fc[1]))
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


