import tensorflow as tf
import numpy as np
##############################
# UPDATE1: Jan 22 2017
# build graph
# a = tf.constant([[1.,4.],[3.,2.]])
# b = tf.constant([[3.,2.],[1.,2.]])
# var1 = tf.Variable(tf.zeros((2,2)),'variable1')
# var2 = tf.Variable(tf.ones((2,2)),'variable2')

a = tf.constant(3.)
b = tf.constant(4.)
var1 = tf.Variable(0.,name = 'variable1')
var2 = tf.Variable(1.,'variable2')
add1 = tf.add(a, var1)
add2 = tf.add(b,var2)
mul = tf.mul(add1,add2)
var1 = tf.assign(var1, mul)
var2 = tf.assign(var2, mul)

# execute

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# print(sess.run([mul,var1,update2,var1]))
	for _ in range(3):
		# sess.run(var1)
		# print(sess.run(update1))
		print(sess.run(var1))

################################


