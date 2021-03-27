import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from tools.utils import normalize_adjacency
from tools.Config import Config
class GraRep(object):
    """
    GraRep Model Object.
    A sparsity aware implementation of GraRep.
    For details see the paper: https://dl.acm.org/citation.cfm?id=2806512
    """
    def __init__(self, graph, dimension, configs = None):
        """
        :param A: Adjacency matrix.
        :param configs: Arguments object.
        """
        self.graph = graph
        self.node2idx = {}  # 节点对应的编号
        self.idx2node = {}
        idx=0
        for node in graph.nodes():
            self.node2idx[node] = idx
            self.idx2node[idx] = node
            idx += 1
        self.A = normalize_adjacency(graph.edges(), self.node2idx)
        
        self._setup_base_target_matrix()
        
        self.configs = Config({
            'output_path' : './output/GraRep_embedding.csv',
            'dimensions' : dimension,
            'order' : 1,
            'seed' : 42,
            'epochs' : 20
        })
        if configs:
            self.configs = self.configs.copy(vars(configs))

    def _setup_base_target_matrix(self):
        """
        Creating a base matrix to multiply.
        创建的是单位矩阵
        """
        values = [1.0 for i in range(self.graph.number_of_nodes())]
        indices = [i for i in range(self.graph.number_of_nodes())]
        self.A_hat = sparse.coo_matrix((values, (indices,indices)),shape=self.A.shape,dtype=np.float32)

    def _create_target_matrix(self):
        """
        Creating a log transformed target matrix.
        :return target_matrix: Matrix to decompose with SVD.
        """
        self.A_hat = sparse.coo_matrix(self.A_hat.dot(self.A))
        scores = np.log(self.A_hat.data)-math.log(self.A.shape[0])
        rows = self.A_hat.row[scores<0]
        cols = self.A_hat.col[scores<0]
        scores = scores[scores<0]
        target_matrix = sparse.coo_matrix((scores, (rows,cols)),shape=self.A.shape,dtype=np.float32)
        return target_matrix

    def train(self):
        """
        Learning an embedding.
        """
        print("\nOptimization started.\n")
        self.embeddings = []
        for step in tqdm(range(self.configs.order)):
            target_matrix = self._create_target_matrix()
            svd = TruncatedSVD(n_components=self.configs.dimensions, n_iter=self.configs.epochs, random_state=self.configs.seed)
            svd.fit(target_matrix)
            embedding = svd.transform(target_matrix)
            self.embeddings.append(embedding)

    def save_embeddings(self, file):
        """
        Saving the embedding.
        """
        print("\nSave embedding.\n")
        embeddings = self.get_embeddings()
        embeddings.to_pickle(file)
        
    def get_embeddings(self,):
        self.embeddings = np.concatenate(self.embeddings,axis=1)
        # column_count = self.configs.order*self.configs.dimensions
        # columns = ["ID"] + ["x_" + str(col) for col in range(column_count)]
        # ids = np.array([i for i in range(self.A.shape[0])]).reshape(-1,1)
        embedding_dict = {}
        for idx in range(self.graph.number_of_nodes()):
            embedding_dict[self.idx2node[idx]] = self.embeddings[idx]
        data = pd.DataFrame.from_dict(embedding_dict, orient ='index')
        #print(data.shape[0])
        return data