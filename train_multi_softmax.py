import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


## Network parameters
batch_size=1000
epochs=100
learning_rate=0.5
dropout_ratio=tf.placeholder(tf.float32)
## Network variables

#layer-1
X=tf.placeholder(tf.float32,[None,784],name="input")
w1=tf.Variable(tf.truncated_normal([784,200],stddev=0.1),name="w1")
b1=tf.Variable(tf.zeros([200])/10,name="b1")
y1=tf.placeholder(tf.float32,[None, 200],name="l1")
y1=tf.nn.relu(tf.matmul(X,w1)+b1)
y1d=tf.nn.dropout(y1,dropout_ratio)
#layer-2
w2=tf.Variable(tf.truncated_normal([200,100],stddev=0.1),name="w2")
b2=tf.Variable(tf.zeros([100])/10,name="b2")
y2=tf.placeholder(tf.float32,[None, 100],name="l2")
y2=tf.nn.relu(tf.matmul(y1d,w2)+b2)
y2d=tf.nn.dropout(y2,dropout_ratio)
#layer-3
w3=tf.Variable(tf.truncated_normal([100,60],stddev=0.1),name="w3")
b3=tf.Variable(tf.zeros([60])/10,name="b3")
y3=tf.placeholder(tf.float32,[None, 60],name="l3")
y3=tf.nn.relu(tf.matmul(y2d,w3)+b3)
y3d=tf.nn.dropout(y3,dropout_ratio)
#layer-4
w4=tf.Variable(tf.truncated_normal([60,30],stddev=0.1),name="w4")
b4=tf.Variable(tf.zeros([30])/10,name="b4")
y4=tf.placeholder(tf.float32,[None, 30],name="l5")
y4=tf.nn.relu(tf.matmul(y3d,w4)+b4)
y4d=tf.nn.dropout(y4,dropout_ratio)
#layer-5
w5=tf.Variable(tf.truncated_normal([30,10],stddev=0.1),name="w5")
b5=tf.Variable(tf.zeros([10]),name="b5")
y5=tf.placeholder(tf.float32,[None, 10],name="l5")
y5=tf.nn.softmax(tf.matmul(y4d,w5)+b5)
#y5d=tf.nn.dropout(y5,dropout_ratio)
## Data placer
y_ = tf.placeholder(tf.float32, [None, 10],name='test')

## Loss function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y5), reduction_indices=[1]))*100
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y5,labels=y_))*100

## optimizer
#optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
optim=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
## training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y5,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for ep in range(epochs):
  batch_count=int(mnist.train.num_examples/batch_size)
  for i in range(batch_count):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optim, feed_dict={X: batch_xs, y_: batch_ys, dropout_ratio:0.85})
  
  print "Epochs: ",ep
  print(" Accuracy: ",accuracy.eval(\
  feed_dict={X: mnist.test.images, y_: mnist.test.labels,dropout_ratio:0.85}))

print "Done"


























