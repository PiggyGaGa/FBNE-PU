import glob
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from tools.Config import Config
#from helper import walk_transformer, create_graph


def walk_transformer(walk, length):
    """
    Tranforming a given random walk to have skips.
    :param walk: Random walk as a list.
    :param length: Skip size.
    :return transformed_walk: Walk chunks for training.
    """
    transformed_walk = []
    for step in range(0,length+1):
        transformed_walk.append([y for i, y in enumerate(walk[step:]) if i % length ==0])
    return transformed_walk

def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class FirstOrderRandomWalker:
    """
    Class to do fast first-order random walks.
    """
    def __init__(self, graph, cfg):
        """
        Constructor for FirstOrderRandomWalker.
        :param graph: Nx graph object.
        :param cfg: Arguments object.
        """
        self.graph = graph
        self.walk_length = cfg.walk_length
        self.walk_number = cfg.walk_number
        self.walks = []

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        for step in range(self.walk_length-1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs)>0:
                walk = walk + random.sample(nebs,1) 
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        """
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.walk_number):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.walk_number)+".\n")
            for node in tqdm(self.graph.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
        return self.walks

class SecondOrderRandomWalker:
    """
    Class to do second-order random walks.
    """    
    def __init__(self, nx_G, is_directed, cfg):
        """
        Constructor for SecondOrderRandomWalker.
        :param  nx_G: Nx graph object.
        :param is_directed: Directed nature of the graph -- True/False.
        :param cfg: Arguments object.
        """
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G.add_weighted_edges_from([(edge[0], edge[1], 1.0), (edge[1], edge[0], 1.0)])
            #self.G[edge[0]][edge[1]]['weight'] = 1.0
            #self.G[edge[1]][edge[0]]['weight'] = 1.0
        self.is_directed = is_directed
        self.walk_length = cfg.walk_length
        self.walk_number = cfg.walk_number
        self.p = cfg.P
        self.q = cfg.Q

    def node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(self.walk_number):
            print("\nRandom walk round: "+str(walk_iter+1)+"/"+str(self.walk_number)+".\n")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.node2vec_walk(start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
             unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
             norm_const = sum(unnormalized_probs)
             normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
             alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return
    
class WalkletMachine:
    """
    Walklet multi-scale graph factorization machine class.
    The graph is being parsed up, random walks are initiated, embeddings are fitted, concatenated and the multi-scale embedding is dumped to disk.
    """
    def __init__(self, graph, dimension, cfg = None):
        """
        Walklet machine constructor.
        :param cfg: Arguments object with the model hyperparameters. 
        """
        if not cfg:
            self.cfg = Config({
                'output' : './output/food_embedding.csv',
                'walk_type' : 'second',
                'total_dimension' : dimension,   #  this not used in algorithm total_dimension = dimensions * window_size
                'dimensions' : 32,
                'walk_number' : 5,
                'walk_length' : 80,
                'window_size' : 4,
                'workers' : 4,
                'min_count' : 1,
                'P' : 1.0,
                'Q' : 1.0})
                
            #print(self.cfg)
        self.graph = graph
        if self.cfg.walk_type == "first":
            self.walker = FirstOrderRandomWalker(self.graph, self.cfg)
        else:
            self.walker = SecondOrderRandomWalker(self.graph, False, self.cfg)
            self.walker.preprocess_transition_probs()
        self.walks = self.walker.do_walks()
        del self.walker

    def train(self):
        self.create_embedding()
    
    def walk_extracts(self, length):
        """
        Extracted walks with skip equal to the length.
        :param length: Length of the skip to be used.
        :return good_walks: The attenuated random walks.
        """
        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_onetime_embeddings(self,):
        """
        Extracting the embedding according to node order from the embedding model.
        :param model: A Word2Vec model after model fitting.
        :return embedding: A numpy array with the embedding sorted by node IDs.
        """
        embedding = {}
        index = 0
        for node in self.graph.nodes():
            embedding[node] = self.Word2vec.wv[node]
            index += 1
        '''
        for node in range(len(self.graph.nodes())):
            embedding.append(list(model[str(node)]))
        embedding = np.array(embedding)
        '''
        embedding = pd.DataFrame.from_dict(embedding, orient ='index')
        #print(embedding.shape[0])
        #print(embedding)
        return embedding

    def create_embedding(self):
        """
        Creating a multi-scale embedding.
        """
        total_embedding = []
        for index in range(1,self.cfg.window_size+1):
            print("\nOptimization round: " +str(index)+"/"+str(self.cfg.window_size)+".")
            print("Creating documents.")
            clean_documents = self.walk_extracts(index)
            print("Fitting model.")
            self.Word2vec = Word2Vec(clean_documents,
                             size = self.cfg.dimensions,
                             window = 1,
                             min_count = self.cfg.min_count,
                             sg = 1,
                             workers = self.cfg.workers)
            new_embedding = self.get_onetime_embeddings()
            #print(new_embedding.shape)
            total_embedding += [new_embedding]
        self.embedding = pd.concat(total_embedding, axis=1)
        #print(self.embedding.shape)
        return 

    def save_embeddings(self, output_file = ''):
        """
        Saving the embedding as a csv with sorted IDs.
        """
        print("\nModels are integrated to be multi scale.\nSaving to disk.")
        #self.column_names = [ "x_" + str(x) for x in range(self.embedding.shape[1])]
        #self.embedding = pd.DataFrame(self.embedding, columns = self.column_names)
        self.embedding.to_pickle(output_file)
    def get_embedding(self):
        return self.embedding