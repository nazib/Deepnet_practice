import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x=tf.placeholder(tf.float32,[None, 784],name='input')
W = tf.Variable(tf.zeros([784, 10]),name='w')
b = tf.Variable(tf.zeros([10]),name='b')
y=tf.placeholder(tf.float32,[None, 10],name='Output')
y=tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10],name='test')


  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
#sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
saver = tf.train.Saver()

#tf.initialize_all_variables()
tf.global_variables_initializer().run()
i=0
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  i=i+1
  print("Iterartion : ",i," Accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)
  
saver.save(sess,'Trained_Soft_max_model/Soft_max_model', global_step=1000)

### Testing the network
