import numpy as np
import tensorflow as tf

a = tf.convert_to_tensor(np.array([[1,2,3,4], [2,5,8,12]]), dtype=tf.float32)
rght = tf.concat((a[..., 1:], tf.expand_dims(a[..., -1], -1)), -1)
left = tf.concat((tf.expand_dims(a[...,0], -1), a[..., :-1]), -1)
ones = tf.ones_like(rght[..., 2:], tf.float32)
one = tf.expand_dims(ones[...,0], -1)
divi = tf.concat((one, ones*2, one), -1)
result = (rght-left) / divi

print(result)