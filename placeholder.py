import tensorflow as tf
import numpy as np
# build graph
a = tf.placeholder(tf.float32,[None,2])
b = tf.placeholder(tf.float32)
mul = tf.matmul(a,b)


# execute
with tf.Session() as sess:
	with tf.device(':/gpu:0'): # did not recognize CPU!!
		sess.run(tf.global_variables_initializer())
		print(sess.run(mul, feed_dict={a:np.array([[1.,2.],[3.,1.]]), b:np.array([[1.,2.],[3.,1.]])}))
		s = sess.run(mul, feed_dict={a:np.array([[1.,2.],[3.,1.]]), b:np.array([[1.,2.],[3.,1.]])})
		print(s)