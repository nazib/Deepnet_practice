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

#### Encoder ####

X=tf.placeholder(tf.float32,[None,784])
inp=tf.reshape(X,[-1,28,28,1])
keepprob=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32,([None,784]))
out=tf.reshape(Y,[-1,28,28,1])


en_layer1=tf.nn.sigmoid(tf.add(conv2d(inp,weights['en_layer1']),bias['en_bias1']))
en_layer1=tf.nn.dropout(en_layer1,keepprob)

en_layer2=tf.nn.sigmoid(tf.add(conv2d(en_layer1,weights['en_layer2']),bias['en_bias2']))
en_layer2=tf.nn.dropout(en_layer2,keepprob)

en_layer3=tf.nn.sigmoid(tf.add(conv2d(en_layer2,weights['en_layer3']),bias['en_bias3']))
en_layer3=tf.nn.dropout(en_layer3,keepprob)


##### Decoder #####
d_layer1=tf.nn.sigmoid(tf.add(conv2d(en_layer3,weights['de_layer1']),bias['de_bias1']))
d_layer1=tf.nn.dropout(d_layer1,keepprob)

d_layer2=tf.nn.sigmoid(tf.add(conv2d(d_layer1,weights['de_layer2']),bias['de_bias2']))
d_layer2=tf.nn.dropout(d_layer2,keepprob)

d_layer3=tf.nn.sigmoid(tf.add(conv2d(d_layer2,weights['de_layer3']),bias['de_bias3']))
d_layer3=tf.nn.dropout(d_layer3,keepprob)

result=d_layer3

######################### Optimization & Loss function ##################
cost=tf.reduce_sum(tf.square(result,-out))
optim=tf.train.AdamOptimizer(0.001).minimize(cost)

########################### Training ################################
batch_size=128
epochs=5
mean_im=np.zeros((784))

with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         
         for epoch_i in range(epochs):
             epoch_loss=0
             for batch_i in range(int(mnist.train.num_examples/batch_size)):
                 batch_x,_=mnist.train.next_batch(batch_size)
                 trainbatch=np.array([im-mean_im for im in batch_x])
                 train_noisy=trainbatch+0.3*np.random.randn(trainbatch.shape[0],784)                             
                 _, c = sess.run([optim, cost], feed_dict={X:train_noisy, Y:trainbatch, keepprob:0.85})
                 epoch_loss += c
             
             
             print('Epoch', epochs, 'completed out of',epoch_i,'loss:',epoch_loss)
        
         print('\n ------------Completed run--------------')
































