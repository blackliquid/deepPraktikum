import tensorflow as tf

class DiscriminatorNet:

    def __init__(self, mean, sdev, batch_size, generatorNet, classify = False):
        self.generatorNet = generatorNet
        self.classify = classify
        self.batch_size = batch_size

        self.defineWeights(mean, sdev)

    def defineWeights(self, mean, sdev):
        with tf.variable_scope("d_scope", initializer=tf.contrib.layers.xavier_initializer()):

            '''conv 0 : batch_size*64*64*3 -> batch_size*64*64*96
             pool 0 : batch_size*64*64*96 -> batch_size*32*32*96
             conv 1 : batch_size*32*32*96 -> batch_size*32*32*192
             pool 1 : batch_size*32*32*192 -> batch_size*16*16*192
             conv 2 : batch_size*16*16*192 -> batch_size*16*16*384
             pool 2 : batch_size*16*16*384 -> batch_size*8*8*384
             conv 3 : batch_size*8*8*384 -> batch_size*8*8*768
             pool 3 : batch_size*8*8*784 -> batch_size*4*4*784
             FC0 : batch_size*4*4*768 -> batch_size*1000
             FC1 : batch_size*1000 -> batch_size*2'''

            # define weights for the convLayers 0-3


            self.W_conv = []
            self.b_conv = []

            # add weight variables for the first convlayer

            #filter dims for conv2d : [h, w, in out]

            self.W_conv.append(tf.get_variable("W_conv_0", [5, 5, 3, 32 * 3]))
            self.b_conv.append(tf.get_variable("b_conv_0", [32 * 3]))

            # number of channels for the last 3 layers

            conv_dims = [96, 192, 384]

            # add weight variables for convlayers 1-3

            for dims, i in zip(conv_dims, range(1,4)):
                self.W_conv.append(tf.get_variable("W_conv_%d" %i, [5, 5, dims, dims * 2]))
                self.b_conv.append(tf.get_variable("b_conv_%d" %i, [dims * 2]))

            # define weights for the two FC layers



            self.W_fc = []
            self.b_fc = []

            # define weights for FC0

            self.W_fc.append(tf.get_variable("W_fc_0", [4 * 4 * 768, 1000]))
            self.b_fc.append(tf.get_variable("b_fc_0", [1000]))

            # define weights for FC1

            self.W_fc.append(tf.get_variable("W_fc_1", [1000, 1]))
            self.b_fc.append(tf.get_variable("b_fc_1", [1]))

            # set reuse flag true!

            tf.get_variable_scope().reuse_variables()

    def defineGraph(self, input_batch):
        #define graph for convlayers
        #array for the results of the conv and pool layers

        self.res_conv = []
        self.res_pool = []

        #define first convlayer as result of the input

        self.res_conv.append(tf.nn.relu(
                tf.nn.conv2d(input_batch, self.W_conv[0], strides=[1, 1, 1, 1], padding='SAME') + self.b_conv[0]))


        #leaky ReLu?
        #self.res_conv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d(self.input_batch, self.W_conv[0], strides=[1, 1, 1, 1], padding='SAME')+self.b_conv[0]))

        self.res_pool.append(tf.nn.avg_pool(self.res_conv[0], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME'))

        #define convlayer 1-3

        for i in range(1, 4):
            self.res_conv.append(tf.nn.relu(tf.nn.conv2d(self.res_pool[i-1], self.W_conv[i], strides=[1, 1, 1, 1], padding='SAME') + self.b_conv[i]))
            #leaky ReLu?
            #self.res_conv.append(tf.contrib.keras.layers.LeakyReLu(tf.nn.conv2d(self.res_pool[i-1], self.W_conv[i], strides=[1, 1, 1, 1], padding='SAME') + self.b_conv[i]))
            self.res_pool.append(tf.nn.avg_pool(self.res_conv[i], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME' ))

        self.res_conv3_flat = tf.reshape(self.res_pool[3], [self.batch_size, 4*4*768])


        #define array for FC layers

        self.res_fc = []

        #fill it with fc0 and fc1

        self.res_fc.append(tf.nn.relu(tf.matmul(self.res_conv3_flat, self.W_fc[0])+self.b_fc[0]))
        #leaky ReLu?
        #self.res_fc.append(tf.contrib.keras.layers.LeakyReLu(tf.matmul(self.res_conv3_flat, self.W_fc[0])+self.b_fc[0]))

        self.res_fc.append(tf.add(tf.matmul(self.res_fc[0], self.W_fc[1]),self.b_fc[1])) #no reLu here!

        #give it a nice name

        self.logits = tf.identity(self.res_fc[1], name="logits")

        #apply sigmoid

        self.probs = tf.nn.sigmoid(self.res_fc[1], name = "probs")

        return self.probs, self.logits