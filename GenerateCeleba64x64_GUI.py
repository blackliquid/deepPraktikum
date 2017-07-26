import tensorflow as tf
import numpy as np
import numpy.random as rnd
import numpy.linalg
import os.path
import scipy.misc
import tkinter as tk
import PIL.Image, PIL.ImageTk

values = np.load('celeba_variables.npz')

sess = tf.InteractiveSession()

def random_filelist(batch_size):
    index = np.random.uniform(1, 202599.99, batch_size)
    index = index.astype(int)
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist

def nums_to_filelist(index):
    filelist = np.array(['%06i.png' % i for i in index])
    return filelist

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


num_fc2 = 512
# Rueckweg
channels = 48  # 8*3RGB also: MUSS DURCH 3 TEILBAR SEIN!

mu = values['mu']
sigma = values['sigma']
# M = np.linalg.cholesky(sigma)
eigenvals, eigenvects = np.linalg.eigh(sigma)

# y_conv_v = np.reshape(np.array(rnd.multivariate_normal(mu, sigma), dtype=np.float32), [1, num_fc2])
y_conv = tf.placeholder(tf.float32, [None, num_fc2])

W_fc1_r = tf.Variable(values['W_fc1_r_v'])
b_fc1_r = tf.Variable(values['b_fc1_r_v'])

h_fc1_r_ = tf.matmul(y_conv, W_fc1_r)
h_fc1_r = tf.add(h_fc1_r_, b_fc1_r)
h_fc1_r_flat = tf.reshape(h_fc1_r, [-1, 16, 16, channels])

W_conv2_r = tf.Variable(values['W_conv2_r_v'])
b_conv2_r = tf.Variable(values['b_conv2_r_v'])
output_shape_conv2r = [1, 32, 32, channels]

h_conv2_r = tf.nn.relu(deconv2d(h_fc1_r_flat, W_conv2_r, output_shape_conv2r) + b_conv2_r)  # deconvolution1

W_conv1_r = tf.Variable(values['W_conv1_r_v'])
b_conv1_r = tf.Variable(values['b_conv1_r_v'])
output_shape_conv1r = [1, 64, 64, channels]

h_conv1_r = deconv2d(h_conv2_r, W_conv1_r, output_shape_conv1r) + b_conv1_r  # deconvolution 2
# output_img = tf.nn.softmax(tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1]),  name='output_img')
# output_img = tf.reshape(tf.reduce_mean(h_conv1_r, axis=3, keep_dims=True), [-1],  name='output_img')
output_img = tf.reshape(h_conv1_r, [1, 64, 64, 3, channels//3])
output_img = tf.reduce_mean(output_img, axis=4, name='output_img')

sess.run(tf.global_variables_initializer())

image_data = None
image_scaled = None
scale = 1

def compute_image(_=None):
    global image_data, image_scaled
    y_start = np.zeros(num_fc2)
    for i in range(num_sliders):
        y_start[num_fc2-i-1] = sliders[i].get()*10000
        # y_start[i] = sliders[i].get()*10000
    y_conv_v = np.array([np.matmul(eigenvects, y_start) + mu])
    img_array = sess.run(output_img, feed_dict={y_conv: y_conv_v})
    image_data = []
    for i in range(64):
        for j in range(64):
            image_data.append(tuple(img_array[0, i, j, :]))
        image_unscaled.putdata(image_data)
    image_scaled = image_unscaled.resize((64*scale, 64*scale))
    image_tk.paste(image_scaled)


def scale_image(newscale):
    global scale, image_tk, image_scaled
    scale = int(newscale)
    canvas.config(width=64*scale, height=64*scale)
    image_scaled = image_unscaled.resize((64*scale, 64*scale))
    image_tk = PIL.ImageTk.PhotoImage(image_scaled)
    canvas.delete('all')
    canvas.create_image(64*scale*0.5, 64*scale*0.5, image=image_tk)


def reset_sliders(_=None):
    for i in sliders:
        i.set(0)


def save_image(_=None):
    number = 0
    while os.path.exists('face%06i.png'%number):
        number += 1
    image_unscaled.save('face%06i.png'%number)


window = tk.Tk()

left = tk.Frame(window)
right = tk.Frame(window)
left.pack(side='left', anchor='nw')
right.pack(side='right')

menuframe = tk.Frame(left)
menuframe.pack(anchor='nw')

button_reset = tk.Button(menuframe, text='reset sliders', command=reset_sliders)
button_reset.pack(anchor='nw')
button_save = tk.Button(menuframe, text='save image', command=save_image)
button_save.pack(anchor='nw')
slider_size = tk.Scale(menuframe, from_=1, to=10, orient='horizontal', command=scale_image)
slider_size.pack(anchor='nw')

canvas = tk.Canvas(left, width=64, height=64)
canvas.pack()
image_unscaled = PIL.Image.new('RGB', (64*scale, 64*scale))
image_tk = PIL.ImageTk.PhotoImage(image_unscaled)
canvas.create_image(64*scale*0.5, 64*scale*0.5, image=image_tk)

sliders = []
num_sliders = 15
for i in range(num_sliders):
    sliders.append(tk.Scale(right, from_=-1, to=1, orient='horizontal', resolution=0.001, command=compute_image))
    sliders[i].pack()

window.mainloop()
