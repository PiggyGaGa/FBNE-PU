import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dpgcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, W):
        x = F.relu(self.gc1(x, W))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, W)
        return F.log_softmax(x, dim=1)
    def get_embeddings(self, x, W, idx2node):
        tensor_embedding =  F.relu(self.gc1(x, W))
        np_embedding = tensor_embedding.detach().numpy()
        embedding = {}
        for idx in sorted(idx2node.keys()):
            embedding[idx2node[idx]] = np_embedding[idx]
        data = pd.DataFrame.from_dict(embedding, orient ='index')
        return data

