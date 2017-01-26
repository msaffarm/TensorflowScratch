import tensorflow as tf

# random seed
# tf.set_random_seed(10) # remove to get different results in each session

# Normal Nomral dist !! actually it may contain outliers(Z_score > 2)
a = tf.random_normal([3,5],mean = 0,stddev= 10) # its stddev instead of std !
b = tf.truncated_normal([1,5], mean = 0, stddev = 10) # avoid outliers
u = tf.random_uniform([2,2], minval =0, maxval= 10)
aShuff = tf.random_shuffle(a) # randomly shuffle rows of a
samples = tf.multinomial(tf.log([[0., 1., 2.]]), 5) # probablities(non-normalized) and sampleSize

with tf.Session() as sess:
	print(sess.run([a,aShuff]))
	print(aShuff.get_shape())# shape of aShuff
	print(sess.run([samples]))