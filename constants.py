import tensorflow as tf
import numpy as np

bnp = np.array([[1,4],[3,8]])
# 2 types of constant
c1 = tf.constant([[1,3,2],[5,8,4]])
c2 = tf.constant(-1.1, shape=[5,2],dtype= 'float32')
# ones_like and zeros_like
zs = tf.zeros_like(bnp)
#
a = tf.fill([2,4],9.) # tf.fill example
b = tf.placeholder(tf.float64,[None, None])
with tf.Session() as sess:
	print(sess.run([a,c1,c2,zs]))
	print(sess.run([b], feed_dict={b:bnp}))