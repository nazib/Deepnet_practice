import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  weight=tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(weight)
  
def bias_variable(shape):
  bias=tf.constant(0.01,shape=shape)
  return tf.Variable(bias)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


####### loading Data #### 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


############## Network Parameters ##############
num_epochs=100
learning_rate=0.5
batch_size=100;

############## Network Architecture #############
hlayer_1=16
hlayer_2=32
hlayer_3=64

inp_layer=784
out_layer=784
inp_image=[-1,28,28,1]

## Filter size ###
ksize=5
imsize=28
############## Convolutional Layers ##############

weights={'en_layer1': weight_variable([ksize,ksize, 1, hlayer_1]),
         'en_layer2': weight_variable([ksize,ksize, hlayer_1, hlayer_2]),
         'en_layer3': weight_variable([ksize,ksize, hlayer_2, hlayer_3]),
##begining decoding layers
         'de_layer1': weight_variable([ksize,ksize, hlayer_3, hlayer_2]),
         'de_layer2': weight_variable([ksize,ksize, hlayer_2, hlayer_1]),
         'de_layer3': weight_variable([ksize,ksize, 1, hlayer_1])}

bias=   {'en_bias1': bias_variable([hlayer_1]),
         'en_bias2': bias_variable([hlayer_2]),
         'en_bias3': bias_variable([hlayer_3]),
##begining decoding layers
         'de_bias1': bias_variable([hlayer_2]),
         'de_bias2': bias_variable([hlayer_1]),
         'de_bias3': bias_variable([1])}

############ Network modeling ################










