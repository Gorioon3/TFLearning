import tensorflow as tf
sess = tf.Session()
x = tf.constant(2)
y = tf.constant(3)
z = tf.constant(7)
print sess.run(x*y+z)
sess.close()
