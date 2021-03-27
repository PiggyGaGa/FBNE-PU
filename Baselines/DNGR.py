#!/usr/bin/env python
# coding: utf-8

"""
Keras implementation of DNGR model. Generate embeddings for NG3, NG6 and NG9   
of 20NewsGroup dataset. Evaluate with F1-score from MNB classifier and NMI score.
Also visualizing embeddings with t-SNE.

Author: Apoorva Vinod Gorur
"""

import sys
import numpy as np
import warnings
import tools.dngr_utils as ut
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE


class DNGR(object):
    def __init__(self, graph, num_hops = 2, alpha_ = 0.98, hidden_neurons = [512, 256, 128]):
        self.graph = graph
        self.idx2node = {}
        index = 0
        for node in graph.nodes():
            self.idx2node[index] = node
            index += 1
        self.num_hops = num_hops
        self.alpha_ = alpha_
        self.hidden_neurons = hidden_neurons

    def train(self,):
        print('begin train model')
        if self.num_hops < 1:
            sys.exit("DNGR: error: argument --hops: Max hops should be a positive whole number")
        if self.alpha_ < 0.0 or self.alpha_ > 1.0:
            sys.exit("DNGR: error: argument --alpha: Alpha's range is 0-1")
        
        if len(self.hidden_neurons) < 3:
            sys.exit("DNGR: error: argument --hidden_neurons: Need a minimum of 3 hidden layers")
        
        #Read groups
        #text_corpus, file_names, target = ut.read_data(group)
        #self.target = target
        #Compute Cosine Similarity Matrix. This acts as Adjacency matrix for the graph.
        #cosine_sim_matrix = ut.get_cosine_sim_matrix(text_corpus)
        Adjacency_matrix = nx.adjacency_matrix(self.graph).todense()
        #Stage 1 - Compute Transition Matrix A by random surfing model
        A = self.random_surf(Adjacency_matrix, self.num_hops, self.alpha_)
        
        #Stage 2 - Compute PPMI matrix 
        PPMI = self.PPMI_matrix(A)

        #Stage 3 - Generate Embeddings using Auto-Encoder
        embeddings = self.sdae(PPMI, self.hidden_neurons)
        self.embeddings = embeddings
        #Evaluation 
        # ut.compute_metrics(embeddings, target)

        # #Visualize embeddings using t-SNE
        # ut.visualize_TSNE(embeddings, target)
        # plt.show()
    def get_embeddings(self):
        new_index = []
        for i in range(self.embeddings.shape[0]):
            new_index.append(self.idx2node[i])
        embeddings = pd.DataFrame(self.embeddings, index=new_index)
        #print(embeddings.shape[0])
        return embeddings
        
    def save_embeddings(self, name):
        #embeddings = self.embeddings.eval()  # tensor to numpy
        # print('embedding result shape {}'.format(self.embeddings.shape))
        new_index = []
        for i in range(self.embeddings.shape[0]):
            new_index.append(self.idx2node[i])
        embeddings = pd.DataFrame(self.embeddings, index=new_index)
        #print(embeddings)
        embeddings.to_pickle(name)
        #pd.DataFrame.to_pickle(name, embeddings)

    #Stage 1 -  Random Surfing
    @ut.timer("Random_Surfing")
    def random_surf(self, cosine_sim_matrix, num_hops, alpha):
        
        num_nodes = len(cosine_sim_matrix)
        
        adj_matrix = ut.scale_sim_matrix(cosine_sim_matrix)
        P0 = np.eye(num_nodes, dtype='float32')
        P = np.eye(num_nodes, dtype='float32')
        A = np.zeros((num_nodes,num_nodes),dtype='float32')
        
        for i in range(num_hops):
            P = (alpha*np.dot(P,adj_matrix)) + ((1-alpha)*P0)
            A = A + P
        return A



    #Stage 2 - PPMI Matrix
    @ut.timer("Generating PPMI matrix")
    def PPMI_matrix(self, A):
        
        num_nodes = len(A)
        A = ut.scale_sim_matrix(A)
        
        row_sum = np.sum(A, axis=1).reshape(num_nodes,1)
        col_sum = np.sum(A, axis=0).reshape(1,num_nodes)
        
        D = np.sum(col_sum)
        PPMI = np.log(np.divide(np.multiply(D,A),np.dot(row_sum,col_sum)))
        #Gotta use numpy for division, else it runs into divide by zero error, now it'll store inf or -inf
        #All Diag elements will have either inf or -inf.
        #Get PPMI by making negative values to 0
        PPMI[np.isinf(PPMI)] = 0.0
        PPMI[np.isneginf(PPMI)] = 0.0
        PPMI[PPMI<0.0] = 0.0
        
        return PPMI

    #Stage 3 - AutoEncoders
    @ut.timer("Generating embeddings with AutoEncoders")
    def sdae(self, PPMI, hidden_neurons):
        
        #local import
        from keras.layers import Input, Dense, noise
        from keras.models import Model

        #Input layer. Corrupt with Gaussian Noise. 
        inp = Input(shape=(PPMI.shape[1],))
        enc = noise.GaussianNoise(0.2)(inp)        
        #Encoding layers. Last layer is the bottle neck
        for neurons in hidden_neurons:
            enc = Dense(neurons, activation = 'relu')(enc)        
        #Decoding layers
        dec = Dense(hidden_neurons[-2], activation = 'relu')(enc)
        for neurons in hidden_neurons[:-3][::-1]:
            dec = Dense(neurons, activation = 'relu')(dec)
        dec = Dense(PPMI.shape[1], activation='relu')(dec)
        
        #Train
        auto_enc = Model(inputs=inp, outputs=dec)
        auto_enc.compile(optimizer='adam', loss='mse')
        
        auto_enc.fit(x=PPMI, y=PPMI, batch_size=10, epochs=5)
        
        encoder = Model(inputs=inp, outputs=enc)
        encoder.compile(optimizer='adam', loss='mse')
        embeddings = encoder.predict(PPMI)
        
        return embeddings



