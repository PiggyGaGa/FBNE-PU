'''
file: main_TaxZ.py
author: PiggyGaGa
email: liana_gyd@163.com
'''

import time
import pandas as pd
import networkx as nx
import community
from Walklets import WalkletMachine
from sdne import SDNE
from line import LINE
from node2vec import Node2Vec
from GraphAttention import GraphAttention
from tools.Config import Config
from GraRep import GraRep
from deepwalk import DeepWalk
from Walklets import WalkletMachine
from DNGR import DNGR
from NetRA import NetRA
from GraphConstruct import GraphConstruct
from tools.Config import Config
from tools.utils import merge_graphs, conn_graphs, get_uniform_node_number_subgraphs
import pymysql


## prepare data, our data is from mysql server  
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
#G = merge_graphs(G)
# 为网络的节点添加属性
for node in G.nodes():
    G.add_node(node, xukai=False)
    
for node in new_xukai:
    G.add_node(node, xukai=True)
print('Trade Network nodes num:{}'.format(len(G.nodes())))
print('Trade Network edges num:{}'.format(len(G.edges())))


global_dimension = 128
cfg = Config({'dimensions' : global_dimension})

DIM=128
name = 'Zhejiang'
time_str = time.strftime(name + '%m-%d-%H:%M:%S',time.localtime(time.time()))

error_file = open('./log/' + time_str + '.log', 'w')
dim_log = open('./log/' + time_str + '.log', 'w')

graph_list = get_uniform_node_number_subgraphs(G)

# print('beging Walklets algorithm ')
# print('\n'*3)
# model = WalkletMachine(G, dimension = global_dimension)
# model.train()
# model.save_embeddings('./Result/' + name + 'walklets_embeddings' + str(DIM) + '.pickle')
# print('Walklets algorithm successful conducted')

print('beging NetRA algorithm ')
print('\n'*3)
total_embeddings = []
for i, g in enumerate(graph_list):
    print('The {} subgraph embedding'.format(i))
    print('node number ', g.number_of_nodes() )
    print('edge number ', g.number_of_edges())
    model = NetRA(g)
    model.train()
    total_embeddings.append(model.get_embeddings())
result = pd.concat(total_embeddings, axis=0)
dim_log.write(name + 'NetRA number instances :' + str(result.shape[0]) + '\n')
result.to_pickle('./Result/' + name + 'netra_embedding' + str(DIM) + '.pickle')
print('NetRA algorithm successful conducted')

# GraRep
try:
    print('beging GraRep algorithm ')
    print('\n'*3)
    total_embeddings = []
    for i, g in enumerate(graph_list):
        print('The {} subgraph embedding'.format(i))
        print('node number ', g.number_of_nodes() )
        print('edge number ', g.number_of_edges())
        model = GraRep(g, global_dimension, cfg)
        model.train()
        total_embeddings.append(model.get_embeddings())
    result = pd.concat(total_embeddings, axis=0)
    dim_log.write(name + 'GraRep number instances :' + str(result.shape[0]) + '\n')
    result.to_pickle('./Result/' + name + 'GraRep_embeddings' + str(DIM) + '.pickle')
    print('GraRep algorithm successful conducted')
except BaseException:
    print('grarep error')

try:
    print('beging NetRA algorithm ')
    print('\n'*3)
    total_embeddings = []
    for i, g in enumerate(graph_list):
        print('The {} subgraph embedding'.format(i))
        print('node number ', g.number_of_nodes() )
        print('edge number ', g.number_of_edges())
        model = NetRA(g)
        model.train()
        total_embeddings.append(model.get_embeddings())
    result = pd.concat(total_embeddings, axis=0)
    dim_log.write(name + 'NetRA number instances :' + str(result.shape[0]) + '\n')
    result.to_pickle('./Result/' + name + 'netra_embedding' + str(DIM) + '.pickle')
    print('NetRA algorithm successful conducted')
except BaseException:
    print('Net RA error ')




try:
# 1. deep walk
    print('beging deep walk algorithm ')
    print('\n'*3)
    model = DeepWalk(G, global_dimension, walk_length=10, num_walks=80, workers=4)
    model.train(window_size=5, workers=4, iter=3)
    model.save_embeddings('./Result/' + name + 'deepwalk_embeddings' + str(DIM) + '.pickle')
    print('deep walk algorithm successful conducted')
except BaseException:
   error_file.write('deep walk algorithm error!\n')
else:
    error_file.write('deep walk algorithm success \n')
# 2. line

try:
    print('beging LINE algorithm ')
    print('\n'*3)
    model = LINE(G, embedding_size=global_dimension, order = 'first')
    model.train(batch_size=64, epochs=50, verbose=2)
    model.save_embeddings('./Result/' + name + 'line_embeddings' + str(DIM) + 'first.pickle')
    print('LINE algorithm successful conducted')

    print('beging LINE algorithm ')
    print('\n'*3)
    model = LINE(G, embedding_size=global_dimension, order = 'second')
    model.train(batch_size=64, epochs=50, verbose=2)
    model.save_embeddings('./Result/' + name + 'line_embeddings' + str(DIM) + 'second.pickle')
    print('LINE algorithm successful conducted')

    print('beging LINE algorithm ')
    print('\n'*3)
    model = LINE(G, embedding_size=global_dimension, order = 'all')
    model.train(batch_size=64, epochs=50, verbose=2)
    model.save_embeddings('./Result/' + name + 'line_embeddings' + str(DIM) + 'both.pickle')
    print('LINE algorithm successful conducted')
except BaseException:
    error_file.write('Line algorithm error\n')
else:
    error_file.write('Line algorithm success \n')

# 3. node2vec
try:
    print('beging Node2Vec algorithm ')
    print('\n'*3)
    model = Node2Vec(G, dimensions=128, walk_length=30,num_walks=20, workers=4)
    embeded = model.fit(window=10, min_count=1, batch_words=4)
    node2vec_dict = {}
    for node in G.nodes():
        node2vec_dict[node] = embeded.wv[node]
    node2vec_pd = pd.DataFrame.from_dict(node2vec_dict, orient='index')
    # model.save_embeddings('./Result/' + name + 'node2vec_embeddings.pickle')
    node2vec_pd.to_pickle('./Result/' + name + 'node2vec_embedding' + str(DIM) + '.pickle')
    print('Node2Vec algorithm successful conducted')
except BaseException:
   error_file.write('node2vec algorithm error!\n')
else:
    error_file.write('node2vec algorithm success \n')


# graph attention

try:
    print('beging graph attention algorithm ')
    print('\n'*3)
    model = GraphAttention(G, global_dimension, cfg)
    model.train()
    model.save_embedding('./Result/' + name + 'graph_attention_embedding' + str(DIM) + '.pickle')
    print('Graph Attention algorithm successful conducted')
except BaseException:
   error_file.write('graph attention algorithm error!\n')
else:
    error_file.write('graph attention algorithm success \n')


# walk lets
try:
    print('beging Walklets algorithm ')
    print('\n'*3)
    model = WalkletMachine(G, dimension = global_dimension)
    model.train()
    model.save_embeddings('./Result/' + name + 'walklets_embeddings' + str(DIM) + '.pickle')
    print('Walklets algorithm successful conducted')
except BaseException:
   error_file.write('walklets algorithm error!\n')
else:
    error_file.write('walklets algorithm success \n')

# grarep
try:
    print('beging GraRep algorithm ')
    print('\n'*3)
    model = GraRep(G, global_dimension, cfg)
    model.train()
    model.save_embeddings('./Result/' + name + 'GraRep_embeddings' + str(DIM) + '.pickle')
    print('GraRep algorithm successful conducted')
except BaseException:
   error_file.write('GraRep algorithm error!\n')
else:
    error_file.write('GraRep algorithm success \n')

# sdne
try:
    print('beging SDNE algorithm ')
    print('\n'*3)
    model = SDNE(G, hidden_size=[256, global_dimension])
    model.train(batch_size=64, epochs=20, verbose=2)
    model.save_embeddings('./Result/' + name + 'sdne_embeddings' + str(DIM) + '.pickle')
    print('SDNE algorithm successful conducted')
except BaseException:
   error_file.write('SDNE algorithm error!\n')
else:
    error_file.write('SDNE algorithm success \n')

# DNGR
try:
    print('beging DNGR algorithm ')
    print('\n'*3)
    model = DNGR(G, hidden_neurons=[256, 128, global_dimension])
    model.train()
    model.save_embeddings('./Result/' + name + 'dngr_embedding' + str(DIM) + '.pickle')
    print('DNGR algorithm successful conducted')
except BaseException:
   error_file.write('DNGR algorithm error!\n')
else:
    error_file.write('DNGR algorithm success \n')


# netra

try:   
    print('beging NetRA algorithm ')
    print('\n'*3)
    model = NetRA(G)
    model.train()
    model.save_embeddings('./Result/' + name + 'netra_embedding' + str(DIM) + '.pickle')
    print('NetRA algorithm successful conducted')
except BaseException:
   error_file.write('NetRA algorithm error!\n')
else:
    error_file.write('NetRA algorithm success \n')


error_file.close()

