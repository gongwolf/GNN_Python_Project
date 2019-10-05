# %%
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import time

from models import GCN, MLP



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Settings
# flags = tf.app.
# FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('hidden1', 24, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay_1', 5e-5, 'Weight for L2 loss on embedding matrix.') # the lambda in the L2-norm
# flags.DEFINE_float('weight_decay_2', 5e-5, 'Weight for L2 loss on embedding matrix.') # the lambda in the L2-norm
# flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

dataset = 'cora'

# %%
# load the data
from utils import *

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    dataset)

#%%
features = preprocess_features(features)
support = [preprocess_adj(adj)]  # L = D^-1/2.A.D^-1/2, It's a list,
print(adj.shape)
# num_supports = 1
# model_func = GCN

# print(support[0])

# placeholders = {
#     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
#     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
# }

print(features[2],y_train.shape[1])

# # Create model
model = GCN(placeholders, input_dim=features[2][1], logging=True)

# sess = tf.Session()


# def evaluate(features, support, labels, mask, placeholders):
#     t_test = time.time()
#     feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
#     outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
#     return outs_val[0], outs_val[1], (time.time() - t_test)


# sess.run(tf.global_variables_initializer())

# cost_val = []

# # Train model
# for epoch in range(FLAGS.epochs):

#     t = time.time()
#     # Construct feed dictionary
#     feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})

#     # Training step
#     outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

#     # Validation
#     cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
#     cost_val.append(cost)

#     # Print results
#     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
#           "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
#           "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

#     if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#         print("Early stopping...")
#         break

# print("Optimization Finished!")

# Testing
# test_cost, test_acc, test_duration = evaluate(
#     features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()


"""Test code of preprocess the adj data"""
# indptr = np.array([0, 2, 5,7,10,13,14])
# indices = np.array([1, 4, 0, 2, 4, 1,3,2,4,5,0,1,3,3])
# data = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# adj = sp.csr_matrix((data, indices, indptr),shape=(6, 6))

# print(adj.toarray())
# print(adj+sp.eye(adj.shape[0]).toarray())

# adj = adj+sp.eye(adj.shape[0])

# adj = sp.coo_matrix(adj)
# rowsum = np.array(adj.sum(1))
# print(rowsum)
# d_inv_sqrt = np.power(rowsum, -0.5).flatten()
# # print(d_inv_sqrt)
# d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
# print(d_mat_inv_sqrt.toarray())
# print(adj.toarray())
# print(adj.dot(d_mat_inv_sqrt).toarray())
# print(adj.dot(d_mat_inv_sqrt).transpose().tocoo().toarray())


"""Test code of dot production"""
# A = sp.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# v = sp.csr_matrix([[1, 0, -1],[3, 0, -1],[3, 0, -3]])
# x = A.dot(v).toarray()
# print(A.dot(v).toarray())
"""Test code for the vstack() and the tolil() function"""
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# a=sp.csr_matrix((data, (row, col)), shape=(3, 3))
# print(a.toarray())


# row = np.array([0, 1, 1, 2, 2, 2])
# col = np.array([0, 1, 2, 0, 1, 2])
# data = np.array([9, 8, 11, 21, 33, 45])
# b=sp.csr_matrix((data, (row, col)), shape=(3, 3))
# print(b.toarray())

# features = sp.vstack((a,b)).tolil()
# print(features.toarray())

"""Test code of the meaning of the sentance """
# index = [3,1,2]
# range_index = np.sort(index)
# print(index,"    ",range_index)
# features[index, :] = features[range_index, :] ##
# print(features.toarray())


# %%
