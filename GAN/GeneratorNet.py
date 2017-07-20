import tensorflow as tf

class GeneratorNet:

    def __init__(self, sdev, batch_size):
        self.sdev = sdev
        self.batch_size = batch_size

        self.defineWeights()
        self.defineGraph()
        self.defineLoss()


    def weight_variable(self, shape):
        # shortcuts for defining the filters

        initial = tf.truncated_normal(shape, stddev=self.sdev)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(self.sdev, shape=shape)
        return tf.Variable(initial)

    def defineWeights(self):
        # define all the weights for the network

        self.W_fc = self.weight_variable([100, 1024*4*4])
        self.b_fc = self.bias_variable([1024*4*4])

        self.W_deconv =  []
        self.b_deconv = []
        deconv_dims = [1024, 512, 256]

        for i in deconv_dims:
            self.W_deconv.append(self.weight_variable(shape=[5, 5, int(i/2), i])) #[x,y,out,in]
            self.b_deconv.append(self.bias_variable([int(i/2)]))

        self.W_deconv.append(self.weight_variable([5, 5, 3, 128]))
        self.b_deconv.append(self.bias_variable([3]))

    def defineGraph(self):
        # create the tf computation graph

        '''FC : 100 -> 1024*4*4
        Deconv 0 : 1024*4*4 -> 512*8*8
        Deconv 1 : 512*8*8 -> 256*16*16
        Deconv 2 : 256*16*16 -> 128*32*32
        Deconv 3 128*32*32 -> 3*64*64'''

        # create uniform random vals als input layer

        self.randVals = tf.random_uniform(shape=[100], minval=0, maxval=1, dtype=tf.float32)
        self.randVals_tensor = tf.reshape(self.randVals, [-1, 100])

        # result of the FC layer

        self.res_fc = tf.add(tf.matmul(self.randVals_tensor, self.W_fc), self.b_fc)
        self.res_fc_tensor = tf.reshape(self.res_fc, [-1, 4, 4, 1024])

        # array for the results and output_shape of the Deconv layers

        self.res_deconv = []
        strides = [1, 2, 2, 1]
        output_shape = [[self.batch_size, 8, 8, 512],[self.batch_size, 16, 16, 256],[self.batch_size, 8, 8, 128], [self.batch_size, 64, 64,3]]

        # first deconv layer 0 with fc as input

        self.res_deconv.append(tf.nn.relu(tf.nn.conv2d_transpose(self.res_fc_tensor, self.W_deconv[0], output_shape[0], strides=strides, padding="VALID")+self.b_deconv[0]))

        # deconv layers 1-2

        for i in range(1, 3):
            self.res_deconv.append(tf.nn.relu(tf.nn.conv2d_transpose(self.res_deconv[i-1], self.W_deconv[i], output_shape[i], strides, padding="VALID")+self.b_deconv[i]))

        # deconv layer 3 = output layer. No Relu here!!

        self.res_deconv.append(tf.add(tf.nn.conv2d_transpose(self.res_deconv[2], self.W_deconv[3], output_shape[3], strides=strides, padding="VALID"),self.b_deconv[3], name="generated_img"))

    def defineLoss(self):
        # define loss here
        pass