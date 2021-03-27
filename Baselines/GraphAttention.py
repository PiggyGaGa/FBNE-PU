import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tools.utils import read_graph, feature_calculator, adjacency_opposite_calculator
from tools.Config import Config

class AttentionWalkLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see: https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention
    """
    def __init__(self, configs = None, shapes = None):
        """
        Setting up the layer.
        :param configs: Arguments object.
        :param shapes: Shape of the target tensor.
        """
        super(AttentionWalkLayer, self).__init__()
        if configs:
            self.configs = configs
        else:
            self.configs = Config({
                'output' : './output/food_embedding.csv',
                'embedding_path' : './output/Graph_attention_embedding.csv',
                'attention_path' : './output/Graph_attention_attention.csv',
                'dimensions': 128,
                'epochs' : 200,
                'window_size' : 5,
                'num_of_walks' : 80,
                'beta' : 0.5,
                'gamma' : 0.5,
                'learning_rate' : 0.001})
        self.shapes = shapes
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], int(self.configs.dimensions/2)))
        self.right_factors = torch.nn.Parameter(torch.Tensor(int(self.configs.dimensions/2),self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0],1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors,-0.01,0.01)
        torch.nn.init.uniform_(self.right_factors,-0.01,0.01)
        torch.nn.init.uniform_(self.attention,-0.01,0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim = 0)
        weighted_target_tensor = weighted_target_tensor * self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_matrix = torch.sum(weighted_target_tensor, dim=0).view(self.shapes[1],self.shapes[2])
        loss_on_target = - weighted_target_matrix * torch.log(torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_opposite = - adjacency_opposite * torch.log(1-torch.sigmoid(torch.mm(self.left_factors, self.right_factors)))
        loss_on_matrices = torch.mean(torch.abs(self.configs.num_of_walks*weighted_target_matrix.shape[0]*loss_on_target + loss_opposite))
        norms = torch.mean(torch.abs(self.left_factors))+torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.configs.beta * (self.attention.norm(2)**2)
        loss = loss_on_matrices +  loss_on_regularization + self.configs.gamma*norms
        return loss
        
class GraphAttention(object):
    '''
    Class for training the AttentionWalk model.
    '''
    def __init__(self, graph, dimension, configs = None):
        """
        Initializing the training object.
        :param configs: Arguments object.
        """
        self.configs = Config({
            'output' : './output/food_embedding.csv',
            'dimensions': dimension,
            'epochs' : 200,
            'window_size' : 5,
            'num_of_walks' : 80,
            'gamma' : 0.5,
            'beta' : 0.5,
            'learning_rate' : 0.001})
        if configs:
            self.configs = self.configs.copy(vars(configs))
        #self.graph = read_graph(self.configs.edge_path)
        self.graph = graph
        self.node2idx = {}
        self.idx2node = {}
        index = 0
        for node in graph.nodes():
            self.node2idx[node] = index
            self.idx2node[index] = node
            index += 1
        self.initialize_model_and_features()

    def initialize_model_and_features(self):
        """
        Creating data tensors and factroization model.
        """
        self.target_tensor = feature_calculator(self.configs, self.graph)
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        self.model = AttentionWalkLayer(self.configs, self.target_tensor.shape)

    def train(self):
        """
        Fitting the model
        """
        print("\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        self.epochs = trange(self.configs.epochs, desc="Loss")
        for epoch in self.epochs:
            self.optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            self.optimizer.step()
            self.epochs.set_description("Attention Walk (Loss=%g)" % round(loss.item(),4))

    def save_model(self):
        """
        Saving the embedding and attention vector.
        """
        self.save_embedding()
        self.save_attention()
    def get_embeddings(self):
        left = self.model.left_factors.detach().numpy()
        right = self.model.right_factors.detach().numpy().T
        #indices = np.array([range(len(self.graph))]).reshape(-1,1)
        embedding = np.concatenate([left, right], axis = 1)
        #print('embedding size', embedding.shape[0])
        # columns = ["x_" + str(x) for x in range(self.configs.dimensions)]
        embedding = pd.DataFrame(embedding)
        new_index = [self.idx2node[id] for id in range(embedding.shape[0])]
        embedding.index = [new_index]
        embedding = pd.DataFrame(embedding)
        return embedding
    def save_embedding(self, file = ''):
        """
        Saving the embedding matrices as one unified embedding.
        """
        print("\nSaving the model.\n")
        embedding = self.get_embeddings()
        pd.to_pickle(embedding, file)
        #embedding.to_csv(file)

    def save_attention(self):
        """
        Saving the attention vector.
        """
        attention = self.model.attention_probs.detach().numpy()
        indices = np.array([range(self.configs.window_size)]).reshape(-1,1)
        attention = np.concatenate([indices, attention], axis = 1)
        attention = pd.DataFrame(attention, columns = ["Order","Weight"])
        attention.to_csv(self.configs.attention_path, index = None)
