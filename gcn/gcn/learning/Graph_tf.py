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