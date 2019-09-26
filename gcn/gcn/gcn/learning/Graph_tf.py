from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

"""Store the graph"""
# writer = tf.compat.v1.summary.FileWriter('.')
# writer.add_graph(tf.compat.v1.get_default_graph())
# writer.flush()

sess = tf.Session()
print(sess.run(total))

"""
During a call to tf.Session.run any tf.Tensor only has a single value. 
The result shows a different random value on each call to run, but a consistent value during a single run (out1 and out2 receive the same random input)
"""
vec = tf.random.uniform(shape=(3,2))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1,out2)))
print(sess.run((out2)))