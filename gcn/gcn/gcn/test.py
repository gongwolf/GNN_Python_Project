from utils import *
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from models import GCN, MLP

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
features = preprocess_features(features)
support = [preprocess_adj(adj)]
print(support)
num_supports = 1
model_func = GCN

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

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