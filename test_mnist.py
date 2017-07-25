import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




sess=tf.InteractiveSession()
saver=tf.train.import_meta_graph('Trained_Soft_max_model/Soft_max_model-1000.meta')
#saver.restore(sess,'Trained_Soft_max_model/Soft_max_model-1000')
saver.restore(sess,tf.train.latest_checkpoint('./Trained_Soft_max_model/'))
#print(sess.run('weight:0'))


graph=tf.get_default_graph()
x=w1=graph.get_tensor_by_name("input:0")
w1=graph.get_tensor_by_name("w:0")
b=graph.get_tensor_by_name("b:0")
y=graph.get_tensor_by_name("Output:0")

tf.global_variables_initializer().run()
i=0
for _ in range(10000):
  batch_xs= mnist.test.images[i]
  batch_ys=mnist.test.labels[i]

  batch_xs=np.reshape(batch_xs,(-1,784))
  batch_ys=np.reshape(batch_ys,(-1,10))

  result=sess.run(y, feed_dict={x: batch_xs,y:batch_ys})
  print(sess.run(tf.arg_max(result,1)))
  i=i+1
print("Testing Done")