'''
file: main_gcn_pncgcn_TaxS.py
author: PiggyGaGa
email: liana_gyd@163.com
'''

from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from pncgcn.utils import load_data, accuracy, load_fix_data, load_nx_W_data
from pncgcn.models import PnCGCN
from gcn.models import GCN
from gcn.utils import load_nx_adj_data
from tools.utils import merge_graphs, conn_graphs, get_uniform_node_number_subgraphs
from GraphConstruct import GraphConstruct



def validate(model, features, adj, labels, idx_val):
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))
    return acc_val


def train(epoch, model, optimizer, adj, features, labels, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    print("Train set results:",
          "loss= {:.4f}".format(loss_train.item()),
          "accuracy= {:.4f}".format(acc_train.item()))
    print('{} epoch'.format(epoch))
    print('time: {:.4f}s'.format(time.time() - t))
    return loss_train.item()

def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

np.random.seed(42)
torch.manual_seed(42)
container = GraphConstruct()
zzsfp_all = pd.read_pickle('../data/zzsfp_all.pickle')
edges = container.get_all_edges_from_dataframe(zzsfp_all)
G = container.construct_network(edges.values)


print('Trade Network nodes num:{}'.format(len(G.nodes())))
print('Trade Network edges num:{}'.format(len(G.edges())))
# graph_list = get_uniform_node_number_subgraphs(G)
label_file = './label/Cluster_label/TaxS_label.pickle'
basic_file = './BasicFeature/Normalize_TaxS_basic.pickle'
epochs = 200
lr = 0.001
weight_decay = 5e-4



# Train model
if __name__ == '__main__':
    # PnCGCN
    name = 'TaxS'
    t_total = time.time()
    # adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_adj_data(G, label_file, exist_fea=basic_file)
    adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_adj_data(G, label_file, exist_fea=basic_file)
    model = PnCGCN(nfeat=features.shape[1], nhid=36, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []
    best_acc = 0
    for epoch in range(epochs):
        loss = train(epoch, model, optimizer, adj, features, labels, idx_train)
        
        acc_val = validate(model, features, adj, labels, idx_val)
        if acc_val > best_acc:
            best_acc = acc_val
        print('best val accuracy {}'.format(best_acc))
        print(' -- '* 5)
        print('\n\n')
        train_loss.append(loss)
    test(model, features, adj, labels, idx_test)
    plt.figure()
    plt.plot(train_loss)
    plt.savefig('./train_loss.png')



    #GCN

    adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_adj_data(G, label_file, exist_fea=basic_file)
    model = GCN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    embedding = model.get_embeddings(features, adj, idx2node)
    for epoch in range(epochs):
        train(epoch, model, optimizer, adj, features, labels, idx_train)
    embedding = model.get_embeddings(features, adj, idx2node)
    print(" Optimization Finished ! ")
    embedding.to_pickle('./Result/' + name + 'GCN_embedding' + str(DIM) + '.pickle')
    
    
    
    
    
