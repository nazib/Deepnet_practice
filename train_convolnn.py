import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



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

###################### Network parameters
batch_size=20
epochs=1000
learning_rate=0.5
dropout_ratio=tf.placeholder(tf.float32)  
  
  

###################### Convolution layers #############################

##### layer-1
X=tf.placeholder(tf.float32,[None,784],name="input")
X_image=tf.reshape(X,[-1,28,28,1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##### layer-2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#### fully connected layer-1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#### drop out
dropout_ratio = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout_ratio)

#### fully connected layer-2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None, 10],name='test')

#### Network optimizer
cross_entropy =tf.reduce_mean(\
tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))*100
                              
optim=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

###################### Training ###########################
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(epochs):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          X: batch[0], y_: batch[1], dropout_ratio: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    optim.run(feed_dict={X: batch[0], y_: batch[1], dropout_ratio: 0.5})















