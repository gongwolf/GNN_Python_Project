#%%
from tensorflow.keras import layers
import tensorflow as tf

class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[1].value, self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)
    super(Linear, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

x = tf.ones((64, 64))
print(x)
linear_layer = Linear(32)
y = linear_layer(x)
print(y)

#%%
# Let's assume we are reusing the Linear class
# with a `build` method that we defined above.

class MLPBlock(layers.Layer):

  def __init__(self):
    super(MLPBlock, self).__init__()
    self.linear_1 = Linear(64)
    self.linear_2 = Linear(64)
    self.linear_3 = Linear(1)

  def call(self, inputs):
    x = self.linear_1(inputs)
    x = tf.nn.relu(x)
    x = self.linear_2(x)
    x = tf.nn.relu(x)
    return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 99)))  # The first call to the `mlp` will create the weights
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))

#%%
# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(layers.Layer):

  def __init__(self, rate=1e-2):
    super(ActivityRegularizationLayer, self).__init__()
    self.rate = rate

  def call(self, inputs):
    self.add_loss(self.rate * tf.reduce_sum(inputs))
    return inputs
    

class OuterLayer(layers.Layer):
  def __init__(self):
    super(OuterLayer, self).__init__()
    # self.activity_reg = ActivityRegularizationLayer(1e-2)
    self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

  def call(self, inputs):
    return self.activity_reg(inputs)

layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)

# layer = OuterLayer()
# assert len(layer.losses) == 0  # No losses yet since the layer has never been called
# _ = layer(tf.zeros(1, 1))
# assert len(layer.losses) == 1  # We created one loss value

# # `layer.losses` gets reset at the start of each __call__
# _ = layer(tf.zeros(1, 1))
# print(len(layer.losses))  # This is the loss created during the call above

# _ = layer(tf.zeros(1, 1))
# print(len(layer.losses))  # This is the loss created during the call above

# print(layer.losses)
#%%
