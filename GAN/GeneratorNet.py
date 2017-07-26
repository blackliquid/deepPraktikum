import tensorflow as tf

class GeneratorNet:

    def __init__(self, mean, sdev, batch_size):
        self.batch_size = batch_size

        self.defineWeights(mean, sdev)
        self.defineGraph()

    def defineWeights(self, mean, sdev):
        with tf.variable_scope("g_scope", initializer=tf.random_normal_initializer(mean, sdev)):
            # define all the weights for the network

            self.W_fc = tf.get_variable("W_fc", [100, 1024 * 4 * 4])
            self.b_fc = tf.get_variable("b_fc", [1024 * 4 * 4])

            self.W_deconv = []
            self.b_deconv = []
            deconv_dims = [512, 256, 128]

            for dims, i in zip(deconv_dims, range(0,3)):
                self.W_deconv.append(tf.get_variable("W_deconv_%d" %i, [5, 5, dims, dims * 2]))  # [x,y,out,in]
                self.b_deconv.append(tf.get_variable("b_deconv_%d" %i, [dims]))

            self.W_deconv.append(tf.get_variable("W_deconv_3", [5, 5, 3, 128]))
            self.b_deconv.append(tf.get_variable("b_deconv_3", [3]))

    def defineGraph(self):
        # create the tf computation graph

        '''FC : 100 -> batch_size*4*4*1024
        Deconv 0 : batch_size*4*4*1024 -> batch_size*8*8*512
        Deconv 1 : batch_size*8*8*512 -> batch_size*16*16*256
        Deconv 2 : batch_size*16*16*256 -> batch_size*32*32*128
        Deconv 3 batch_size*32*32*128 -> batch_size*64*64*3'''

        # create uniform random vals als input layer

        self.randVals = tf.random_uniform(shape=[100], minval=0, maxval=1, dtype=tf.float32)
        self.randVals_tensor = tf.reshape(self.randVals, [-1, 100])

        # result of the FC layer

        self.res_fc = tf.nn.relu(tf.add(tf.matmul(self.randVals_tensor, self.W_fc), self.b_fc))
        #leaky ReLu?
        ##self.res_fc = tf.contrib.keras.layers.LeakyReLu(tf.add(tf.matmul(self.randVals_tensor, self.W_fc), self.b_fc))
        self.res_fc_tensor = tf.reshape(self.res_fc, [-1, 4, 4, 1024])

        # array for the results and output_shape of the Deconv layers

        self.res_deconv = []
        strides = [1, 2, 2, 1]
        output_shape = [[self.batch_size, 8, 8, 512],[self.batch_size, 16, 16, 256],[self.batch_size, 32, 32, 128], [self.batch_size, 64, 64,3]]

        # first deconv layer 0 with fc as input
        self.res_deconv.append(tf.nn.relu(tf.nn.conv2d_transpose(self.res_fc_tensor, self.W_deconv[0], output_shape[0], strides=strides, padding="SAME")+self.b_deconv[0]))

        #leaky ReLu?
        #self.res_deconv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d_transpose(self.res_fc_tensor, self.W_deconv[0], output_shape[0], strides=strides, padding="VALID")+self.b_deconv[0]))

        # deconv layers 1-2

        for i in range(1, 3):
            self.res_deconv.append(tf.nn.relu(tf.nn.conv2d_transpose(self.res_deconv[i-1], self.W_deconv[i], output_shape[i], strides, padding="SAME")+self.b_deconv[i]))
            #leaky ReLu?
            #self.res_deconv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d_transpose(self.res_deconv[i-1], self.W_deconv[i], output_shape[i], strides, padding="VALID")+self.b_deconv[i]))

        # deconv layer 3 = output layer. No Relu here!!

        self.res_deconv.append(tf.add(tf.nn.conv2d_transpose(self.res_deconv[2], self.W_deconv[3], output_shape[3], strides=strides, padding="SAME"),self.b_deconv[3]))

        # give it a nice name

        self.generated_img = tf.nn.sigmoid(self.res_deconv[3], name="generated_img")