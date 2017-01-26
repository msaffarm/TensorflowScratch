import tensorflow as tf

# 2 simple kind of sequences in tf
a = tf.linspace(10.,100.,10) # start and stop in {float32,float64}, num in {int32,int64}
b = tf.range(10.)
c = tf.range(1,10,2)

with tf.Session() as sess:
	print(sess.run([a,b,c]))