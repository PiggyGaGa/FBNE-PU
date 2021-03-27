import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import networkx as nx
import pandas as pd
from sklearn.preprocessing import normalize as sknormalize


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora", if_feat=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    if if_feat:
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = normalize(features)
    else:
        features = sp.csr_matrix(np.identity(idx_features_labels.shape[0]))
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    features = np.identity(features.size(0))
    features = torch.FloatTensor(features)
    return adj, features, labels, idx_train, idx_val, idx_test

def load_fix_data(path="./data/cora/", dataset="cora", if_feat=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    if if_feat:
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = normalize(features)
    else:
        features = sp.csr_matrix(np.identity(idx_features_labels.shape[0]))
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #print(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    beta_par = 0.01
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    degrees = adj.sum(1)
    C_p = adj + adj.T * adj - csr_matrix(np.diag((adj.T * adj).diagonal()))
    #print(C_pi)
    degrees = np.array(degrees).flatten()
    indics = [ i for i, node in enumerate(range(degrees.size))]
    D = csr_matrix((degrees ** (-0.5), (indics, indics)), shape = (degrees.size, degrees.size))
    D1 = csr_matrix((degrees ** (-beta_par), (indics, indics)), shape = (degrees.size, degrees.size))
    D2 = csr_matrix((degrees ** (-beta_par), (indics, indics)), shape = (degrees.size, degrees.size))
    M = D1.T * C_p * D2
    M = D * M * D
    #M = M + M.T.multiply(M.T > M) - M.multiply(M.T > M)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    M = sparse_mx_to_torch_sparse_tensor(M)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return M, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape) 
    return torch.sparse.FloatTensor(indices, values, shape)

def get_idx(labels):
    positive_index =np.where(labels == 1)[0]
    negative_index = np.where(labels == 0)[0]
    pos_length = positive_index.size
    neg_length = negative_index.size
    pos_train = positive_index[: pos_length // 5]
    pos_val = positive_index[pos_length // 5 : int(pos_length * 2 / 5)]
    pos_test = positive_index[int(pos_length * 2 / 5):]

    neg_train = negative_index[: 1000]
    neg_val = negative_index[1000:3000]
    neg_test = negative_index[3000:10000]

    train_idx = np.concatenate((pos_train, neg_train), axis=0)
    val_idx = np.concatenate((pos_val, neg_val), axis=0)
    test_idx = np.concatenate((pos_test, neg_test), axis=0)

    return train_idx, val_idx, test_idx

def load_nx_W_data(nx_graph, label_file, exist_fea = False):
    label_dataframe = pd.read_pickle(label_file)
    
    beta_par = 1
    idx2node = { i: node for i, node in enumerate(nx_graph.nodes())}
    #node2idx = {node:idx for idx, node in enumerate(nx_graph.nodes())}
    n_nodes = nx_graph.number_of_nodes()

    labels = get_label(idx2node, label_dataframe)
    # get adjacency matrix
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # https://github.com/tkipf/pygcn/issues/3 explain
    # get M
    C_p = adj + adj.T * adj - csr_matrix(np.diag((adj.T * adj).diagonal()))
    degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
    degrees = np.array(degrees, dtype=np.float)
    indics = [ i for i, node in enumerate(range(degrees.size))]
    D = csr_matrix((degrees ** (-0.5), (indics, indics)), shape = (degrees.size, degrees.size))
    D1 = csr_matrix((degrees ** (-beta_par), (indics, indics)), shape = (degrees.size, degrees.size))
    D2 = csr_matrix((degrees ** (-beta_par), (indics, indics)), shape = (degrees.size, degrees.size))
    M = D1.T * C_p * D2
    M = D * M * D
    #M = adj

    # get labels
    
    if exist_fea:
        features = get_features(idx2node, exist_fea)
        print('basic features shape', features.shape)
    else:
        features = np.identity(n_nodes)
    features = sknormalize(features, axis = 1)

    # get idx
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    idx_train, idx_val, idx_test = get_idx(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    M = sparse_mx_to_torch_sparse_tensor(M)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return M, features, labels, idx_train, idx_val, idx_test, idx2node



def load_nx_adj_data(nx_graph, label_file, exist_fea = False):
    label_dataframe = pd.read_pickle(label_file)
    idx2node = { i: node for i, node in enumerate(nx_graph.nodes())}
    #node2idx = {node:idx for idx, node in enumerate(nx_graph.nodes())}
    n_nodes = nx_graph.number_of_nodes()
    labels = get_label(idx2node, label_dataframe)
    idx_train, idx_val, idx_test = get_idx(labels)
    print('sum labels', sum(labels))
    # print(labels)
    # get adjacency matrix
    adj = nx.adjacency_matrix(nx_graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # get labels
    if exist_fea:
        features = get_features(idx2node, exist_fea)
        print('basic features shape', features.shape)
    else:
        features = np.identity(n_nodes)


    # get idx
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, idx2node


def get_features(idx2node, basic_features):
    data = pd.read_pickle(basic_features)
    index = sorted(idx2node.keys())
    result = []
    for idx in index:
        fea = data.loc[idx2node[idx]].values.flatten()
        result.append(fea)
    return np.array(result)

def get_label(idx2node, label_dict):
    index = sorted(idx2node.keys())
    result = []
    idx_list = label_dict.index.tolist()
    for idx in index:
        if idx2node[idx] in idx_list:
            label = int(label_dict.loc[idx2node[idx]].values[0])
        else:
            label = 0
        result.append(label)
    return np.array(result)