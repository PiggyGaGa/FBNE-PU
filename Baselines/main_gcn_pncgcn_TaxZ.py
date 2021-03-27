'''
file: main_gcn_pncgcn_TaxZ.py
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

import torch
import torch.nn.functional as F
import torch.optim as optim

from pncgcn.utils import load_data, accuracy, load_fix_data, load_nx_W_data
from pncgcn.models import PnCGCN
from gcn.models import GCN
from gcn.utils import load_nx_adj_data
from tools.utils import merge_graphs, conn_graphs, get_uniform_node_number_subgraphs
from GraphConstruct import GraphConstruct
import pymysql

# Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=True,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=5,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=128,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')

# args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)


connection = pymysql.connect(
    host = '127.0.0.1',
    user = '***', 
    password = '******',
    database = '****',
    port = 3306,
    charset = 'utf8mb4', 
    cursorclass = pymysql.cursors.Cursor)
try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT XFNSRDZDAH, GFNSRDZDAH from ZheJiang.VATInvoice"
        cursor.execute(sql)
        result = cursor.fetchall()
        # 用 'XUJIAFAPIAO' 代替 None
        edges = []
        n=0
        for edge in result:
            edge = list(edge)
            if edge[0] == None:
                edge[0] = 'XUJIAFAPIAO'
                n += 1
            if edge[1] == None:
                edge[1] = 'XUJIAFAPIAO'
                n += 1
            edges.append(edge)
        print('None exist {} times'.format(n))
        result = pd.DataFrame(edges)
        result.drop_duplicates(inplace=True) # remove duplicate
        result = result.values
finally:
    print('get 销方纳税人电子档案号和购方纳税人电子档案号！')
    #connection.close()
try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT NSRDZDAH from ZheJiang.XuKaiList"
        cursor.execute(sql)
        xukai = cursor.fetchall()
        #print(result[0][0])
finally:    
    new_xukai = []
    for item in xukai:
        new_xukai.append(item[0])
    new_xukai.append('XUJIAFAPIAO')
    print('get 虚开企业名单')
    #connection.close()


container = GraphConstruct()
G = container.construct_network(result)




def validate(model, features, adj, labels, idx_val):
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))


def train(epoch, model, optimizer, adj, features, labels, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output, labels)
    loss_train.backward()
    optimizer.step()
    print('{} epoch'.format(epoch))
    print('time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))



# for node in G.nodes():
#     G.add_node(node, xukai=False)
    
# for node in new_xukai:
#     G.add_node(node, xukai=True)
print('Trade Network nodes num:{}'.format(len(G.nodes())))
print('Trade Network edges num:{}'.format(len(G.edges())))
# graph_list = get_uniform_node_number_subgraphs(G)
label_file = '/home/dada/TAX/ShangHai/MGraphEmbedding/label/Zhejiang_label.pickle'
basic_file = './BasicFeature/Normalize_Zhejiang_basic.pickle'
# adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_data(G, label_file, exist_fea=False)
epochs = 100
lr = 0.001
weight_decay = 5e-4
DIM=128

# Train model
# if __name__ == '__main__':

#     t_total = time.time()
#     total_embedding = []
#     for g in graph_list:
#         adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_data(g, label_file, exist_fea=False)
#         model = GCN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1, dropout=0.5)
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         embedding = model.get_embeddings(features, adj, idx2node)
#         for epoch in range(epochs):
#             train(epoch, model, optimizer, adj, features, labels, idx_train)
#         embedding = model.get_embeddings(features, adj, idx2node)
#         total_embedding.append(embedding)
#     print(" Optimization Finished ! ")
#     result = pd.concat(total_embedding, axis=0)
#     result.to_pickle('./Result/' + 'dpgcn' + 'ShangHaiembeddings' + '128' + '.pickle')


# Train model
if __name__ == '__main__':
    # PnCGCN
    name = 'ZheJiang'
    t_total = time.time()
    adj, features, labels, idx_train, idx_val, idx_test , idx2node = load_nx_W_data(G, label_file, exist_fea=basic_file)
    model = PnCGCN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        train(epoch, model, optimizer, adj, features, labels, idx_train)
    embedding = model.get_embeddings(features, adj, idx2node)
    print(" Optimization Finished ! ")
    embedding.to_pickle('./Result/' + name + 'PnCGCN_embedding' + str(DIM) + '.pickle')
    GCNbedding = model.get_embeddings(features, adj, idx2node)
    
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